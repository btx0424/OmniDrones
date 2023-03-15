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
from setproctitle import setproctitle
from tensordict import TensorDict
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs.transforms import TransformedEnv, InitTracker

from tqdm import tqdm


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs import Hover
    from omni_drones.learning.mappo import MAPPOPolicy
    from omni_drones.sensors.camera import Camera

    env = Hover(cfg, headless=cfg.headless)

    agent_spec = env.agent_spec["drone"]
    ppo = MAPPOPolicy(
        cfg.algo, agent_spec=agent_spec, device="cuda"
    )

    def log(info):
        for k, v in info.items():
            print(f"train/{k}: {v}")
        run.log(info)

    logger = LogOnEpisode(
        cfg.env.num_envs,
        in_keys=["return", "progress"],
        log_keys=["train/return", "train/ep_length"],
        logger_func=log,
    )

    env = TransformedEnv(env, InitTracker())
    collector = SyncDataCollector(
        env,
        ppo,
        callback=logger,
        frames_per_batch=env.num_envs * cfg.algo.train_every,
        device=cfg.sim.device,
        return_same_td=True,
    )

    camera = Camera(**env.DEFAULT_CAMERA_CONFIG)

    @torch.no_grad()
    def evaluate():
        info = {"env_frames": collector._frames}
        frames = []

        def record_frame(*args, **kwargs):
            env.sim.render()
            frame = camera()["rgb"].cpu()
            frames.append(frame)

        env.enable_render = True
        env.rollout(
            max_steps=500,
            policy=policy,
            callback=record_frame,
            auto_reset=True,
        )
        env.enable_render = not cfg.headless

        info["recording"] = wandb.Video(
            torch.stack(frames).permute(0, 3, 1, 2), fps=1 / cfg.sim.dt, format="mp4"
        )
        return info

    pbar = tqdm(collector)
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames}
        info.update(ppo.train_op(data))

        if i % 2 == 0:
            run.log(info)

        # if i % 100 == 0:
        #     logging.info(f"Eval at {collector._frames} steps.")
        #     run.log(evaluate())

        pbar.set_postfix({
            "rollout_fps": collector._fps,
            "frames": collector._frames,
            "episodes": collector._episodes,
        })

    simulation_app.close()


if __name__ == "__main__":
    main()
