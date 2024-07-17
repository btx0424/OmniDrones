import os

import hydra
import torch
import wandb

from torch.func import vmap
from tqdm import trange
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


@hydra.main(version_base=None, config_path=".", config_name="demo_task")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))

    run = wandb.init()
    setproctitle(run.name)

    model_artifact = run.use_artifact(f"{cfg.wandb.entity}/{cfg.wandb.project}/{cfg.wandb.artifact_name}:{cfg.wandb.artifact_version}")
    model_dir = model_artifact.download()
    model_filename = "checkpoint_final.pt"
    for key in getattr(model_artifact, "_manifest").entries.keys():
        print(key)
        if key.split('.')[-1] == "pt":
            model_filename = key
    model_path = os.path.join(model_dir, model_filename)
    artifact_config = model_artifact.metadata
    for k, v in artifact_config.items():
        if isinstance(v, str):
            artifact_config[k] = eval(v)
    artifact_cfg = OmegaConf.create(artifact_config)
    artifact_cfg.env.num_envs = cfg.num_envs
    artifact_cfg.task.env.num_envs = cfg.num_envs
    artifact_cfg.env.max_episode_length = cfg.max_episode_length
    artifact_cfg.task.env.max_episode_length = cfg.max_episode_length
    print(OmegaConf.to_yaml(artifact_cfg))

    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[artifact_cfg.task.name]
    base_env = env_class(artifact_cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by default processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if artifact_cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if artifact_cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)
    if (
        artifact_cfg.task.get("ravel_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
        and isinstance(base_env.observation_spec[("agents", "intrinsics")], CompositeSpec)
    ):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1))

    # optionally discretize the action space or use a controller
    action_transform: str = artifact_cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms))
    env.set_seed(artifact_cfg.seed)

    try:
        policy = ALGOS[artifact_cfg.algo.name.lower()](
            artifact_cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )
        policy.load_state_dict(torch.load(model_path))
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {artifact_cfg.algo.name}")

    max_iters = cfg.get("max_iters", -1)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)

    base_env.enable_render(not cfg.headless)
    base_env.eval()
    env.eval()

    @torch.no_grad()
    def evaluate(
        exploration_type: ExplorationType=ExplorationType.MODE
    ):
        render_callback = RenderCallback(interval=2)

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        # log video
        info["recording"] = wandb.Video(
            render_callback.get_video_array(axes="t c h w"),
            fps=0.5 / (artifact_cfg.sim.dt * artifact_cfg.sim.substeps),
            format="mp4"
        )

        # log distributions
        # df = pd.DataFrame(traj_stats)
        # table = wandb.Table(dataframe=df)
        # info["eval/return"] = wandb.plot.histogram(table, "return")
        # info["eval/episode_len"] = wandb.plot.histogram(table, "episode_len")

        return info

    for _ in trange(max_iters):
        info = {}
        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)
        info.update(evaluate())
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
