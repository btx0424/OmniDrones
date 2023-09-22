import logging
import os
import time

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from functorch import vmap
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from torchrl.data import CompositeSpec
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction, 
    FromDiscreteAction,
    ravel_composite,
    VelController,
    AttitudeController,
    History
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.learning import (
    PPOAdaptiveSeparate, PPOAdaptiveJoint, 
    PPORNNPolicy, PPOPolicy,
)

from setproctitle import setproctitle
from torchrl.envs.transforms import (
    TransformedEnv, 
    InitTracker, 
    Compose,
    CatTensors
)

from tqdm import tqdm

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1

from typing import Sequence
from tensordict import TensorDictBase

class EpisodeStats:
    def __init__(self, in_keys: Sequence[str] = None):
        self.in_keys = in_keys
        self._stats = []
        self._episodes = 0

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        done = tensordict.get(("next", "done"))
        truncated = tensordict.get(("next", "truncated"), None)
        done_or_truncated = (
            (done | truncated) if truncated is not None else done.clone()
        )
        if done_or_truncated.any():
            done_or_truncated = done_or_truncated.squeeze(-1)
            self._episodes += done_or_truncated.sum().item()
            self._stats.extend(
                tensordict.select(*self.in_keys)[done_or_truncated].clone().unbind(0)
            )
    
    def pop(self):
        stats: TensorDictBase = torch.stack(self._stats).to_tensordict()
        self._stats.clear()
        return stats

    def __len__(self):
        return len(self._stats)


@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv
    algos = {
        "ppo": PPOPolicy,
        "ppo_adaptive_separate": PPOAdaptiveSeparate,
        "ppo_adaptive_joint": PPOAdaptiveJoint,
        "ppo_rnn": PPORNNPolicy,
    }
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by deafault processed by a entity-based encoder
    # flatten it to use a MLP encoder instead
    if cfg.task.get("flatten_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("flatten_state", False):
        transforms.append(ravel_composite(base_env.observation_spec, "state"))
    if (
        cfg.task.get("flatten_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
        and isinstance(base_env.observation_spec[("agents", "intrinsics")], CompositeSpec)
    ):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1))

    # if cfg.task.get("history", False):
    #     # transforms.append(History([("info", "drone_state"), ("info", "prev_action")]))
    #     transforms.append(History([("agents", "observation")]))

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform == "velocity":
            from omni_drones.controllers import LeePositionController
            controller = LeePositionController(9.81, base_env.drone.params).to(base_env.device)
            transform = VelController(vmap(controller))
            transforms.append(transform)
        elif action_transform == "rate":
            from omni_drones.controllers import RateController
            transform = RateController.make()
            transforms.append()
        elif action_transform == "attitude":
            from omni_drones.controllers import AttitudeController as _AttitudeController
            controller = _AttitudeController(9.81, base_env.drone.params).to(base_env.device)
            transform = AttitudeController(vmap(vmap(controller)))
            transforms.append(transform)
        elif not action_transform.lower() == "none":
            raise NotImplementedError(f"Unknown action transform: {action_transform}")
    
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    policy = algos[cfg.algo.name.lower()](
        cfg.algo, env.observation_spec, env.action_spec, env.reward_spec, device=base_env.device)

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(seed: int=0):
        frames = []

        from tqdm import tqdm
        t = tqdm(total=base_env.max_episode_length)

        def record_frame(*args, **kwargs):
            frame = env.base_env.render(mode="rgb_array")
            frames.append(frame)
            t.update(2)

        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)
        trajs = env.rollout(
            max_steps=base_env.max_episode_length,
            policy=policy,
            callback=Every(record_frame, 2),
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        )
        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        truncated = trajs.get(("next", "truncated"), None)
        done_or_truncated = (done | truncated) if truncated is not None else done
        first_done = torch.argmax(done_or_truncated.long(), dim=1).cpu()

        def take_first(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices)
        
        traj_stats = trajs["next"].select(*stats_keys).cpu().apply(take_first, batch_size=[len(first_done)])
        info = {
            "eval/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v).item() 
            for k, v in traj_stats.items(True, True)
        }
        video_array = np.stack(frames).transpose(0, 3, 1, 2)
        frames.clear()
        info["recording"] = wandb.Video(video_array, fps=0.5 / (cfg.sim.dt * cfg.sim.substeps), format="mp4")
        
        df = pd.DataFrame(traj_stats["stats"].to_dict())
        table = wandb.Table(dataframe=df)
        info["eval/return"] = wandb.plot.histogram(table, "return")
        info["eval/episode_len"] = wandb.plot.histogram(table, "episode_len")
        
        if (
            hasattr(policy, "adaptation_loss_traj")
            and policy.phase == "adaptation"
        ):
            fig, axes = plt.subplots(5, 2, sharex=True)
            for i in range(5):
                traj_loss = policy.adaptation_loss_traj(traj.to(policy.device))
                axes[i, 0].plot(traj_loss["mse"], label="mse")
                axes[i, 0].plot(traj_loss["value_error"], label="value_discrepancy")

            info["eval/adaptation_loss_traj"] = fig
        
        fig, axes = plt.subplots(5, 2, sharex=True)
        for i in range(5):
            traj = trajs[i, :first_done[i].item()]
            # axes[i, 0].set_title("tracking_error")
            # axes[i, 0].plot(traj[("stats", "tracking_error")])
            axes[i, 1].set_title("action")
            axes[i, 1].plot(traj[("agents", "action")].squeeze(1))
        info["eval/traj_logs"] = fig

        return info

    pbar = tqdm(collector)
    env.train()
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats(data.to_tensordict())
        
        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        info.update(policy.train_op(data.to_tensordict()))

        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate())
            env.train()
            base_env.train()

        if save_interval > 0 and i % save_interval == 0:
            try:
                ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                logging.info(f"Saved checkpoint to {str(ckpt_path)}")
            except AttributeError:
                logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        run.log(info)
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({
            "rollout_fps": collector._fps,
            "frames": collector._frames,
        })

        if max_iters > 0 and i >= max_iters - 1:
            break 
    
    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate())
    run.log(info)

    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except AttributeError:
        logging.warning(f"Policy {policy} does not implement `.state_dict()`")
        

    wandb.save(os.path.join(run.dir, "checkpoint*"))
    wandb.finish()
    
    simulation_app.close()


if __name__ == "__main__":
    main()
