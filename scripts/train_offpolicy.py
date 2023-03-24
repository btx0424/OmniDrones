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
from omni_drones.utils.envs.transforms import LogOnEpisode
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.math import quaternion_to_euler

from setproctitle import setproctitle
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
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

    from omni_drones.envs import IsaacEnv
    from omni_drones.learning.sac import MASACPolicy
    from omni_drones.learning.qmix import QMIX
    from omni_drones.sensors.camera import Camera

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    camera = Camera()
    camera.spawn(["/World/Camera"], translations=[(6, 6, 3)], targets=[(0, 0, 1)])
    camera.initialize("/World/Camera")

    agent_spec = base_env.agent_spec["drone"]
    policy = QMIX(
        cfg.algo, agent_spec=agent_spec, device="cuda"
    )

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

    env = TransformedEnv(base_env, Compose(InitTracker(), logger)) 

    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=env.num_envs * cfg.algo.train_every,
        total_frames=-1,
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
        env.rollout(
            max_steps=base_env.max_episode_length,
            policy=policy,
            callback=record_frame,
            auto_reset=True,
            break_when_any_done=False,
        )
        base_env.enable_render(not cfg.headless)

        info["recording"] = wandb.Video(
            torch.stack(frames).permute(0, 3, 1, 2), fps=1 / cfg.sim.dt, format="mp4"
        )
        return info

    pbar = tqdm(collector)
    for i, data in enumerate(pbar):        
        info = {"env_frames": collector._frames}
        info.update(policy.train_op(data))

        # run.log(info)

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
