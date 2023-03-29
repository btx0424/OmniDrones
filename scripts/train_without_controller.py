import logging
import os
import time

import hydra
import torch
import wandb
from functorch import vmap
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.learning.collectors import SyncDataCollector
from omni_drones.utils.envs.transforms import LogOnEpisode, FromMultiDiscreteAction, FromDiscreteAction
from omni_drones.utils.wandb import init_wandb

from setproctitle import setproctitle
from tensordict import TensorDict
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

from tqdm import tqdm


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
    from omni_drones.learning.mappo import MAPPOPolicy
    from omni_drones.sensors.camera import Camera

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    
    def log(info):
        for k, v in info.items():
            print(f"{k}: {v}")
        run.log(info)

    logger = LogOnEpisode(
        cfg.env.num_envs,
        in_keys=["return", "progress"],
        log_keys=["train/return", "train/ep_length"],
        logger_func=log,
    )
    transforms = [InitTracker(), logger]

    if cfg.task.get("multidiscrete_action", None):
        nbins = cfg.task.multidiscrete_action.nbins
        transform = FromMultiDiscreteAction(("action", "drone.action"), nbins=nbins)
        transforms.append(transform)
    elif cfg.task.get("discrete_action", None):
        nbins = cfg.task.discrete_action.nbins
        transform = FromDiscreteAction(("action", "drone.action"), nbins=nbins)
        transforms.append(transform)
    
    env = TransformedEnv(base_env, Compose(*transforms)) 

    camera = Camera()
    camera.spawn(["/World/Camera"], translations=[(6, 6, 3)], targets=[(0, 0, 1)])
    camera.initialize("/World/Camera")

    # TODO: create a agent_spec view for TransformedEnv
    agent_spec = AgentSpec(
        name=base_env.agent_spec["drone"].name,
        n=base_env.agent_spec["drone"].n,
        observation_spec=env.observation_spec["drone.obs"],
        action_spec=env.action_spec["drone.action"],
        reward_spec=env.reward_spec["drone.reward"],
        state_spec=env.observation_spec["drone.state"] if base_env.agent_spec["drone"].state_spec is not None else None,
    )
    policy = MAPPOPolicy(
        cfg.algo, agent_spec=agent_spec, device="cuda"
    )

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate():
        info = {"env_frames": collector._frames}
        frames = []

        def record_frame(*args, **kwargs):
            frame = camera.get_images()["rgb"][0]
            frames.append(frame.cpu())

        base_env.enable_render(True)
        base_env.rollout(
            max_steps=base_env.max_episode_length,
            policy=policy,
            callback=record_frame,
            auto_reset=True,
            break_when_any_done=False,
        )
        base_env.enable_render(not cfg.headless)
        env.reset()

        info["recording"] = wandb.Video(
            torch.stack(frames).permute(0, 3, 1, 2), fps=1 / cfg.sim.dt, format="mp4"
        )
        return info

    pbar = tqdm(collector)
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames}
        info.update(policy.train_op(data))

        run.log(info)

        # if i % 100 == 0:
        #     logging.info(f"Eval at {collector._frames} steps.")
        #     run.log(evaluate())

        pbar.set_postfix({
            "rollout_fps": collector._fps,
            "frames": collector._frames,
        })

    simulation_app.close()


if __name__ == "__main__":
    main()
