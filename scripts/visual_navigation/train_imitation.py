import logging
import os
import time

import hydra
import torch
import wandb
from functorch import vmap
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.envs.transforms import LogOnEpisode
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.math import quaternion_to_euler

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

    from omni_drones.envs import Forest
    from omni_drones.learning.mappo import MAPPOPolicy
    from omni_drones.sensors.camera import Camera

    base_env = Forest(cfg, headless=cfg.headless)
    camera = Camera()
    camera.spawn(["/World/Camera"], translations=[(6, 6, 3)], targets=[(0, 0, 1)])
    camera.initialize("/World/Camera")

    agent_spec = base_env.agent_spec["drone"]
    agent_spec.action_spec = UnboundedContinuousTensorSpec(3, device=base_env.device)
    ppo = MAPPOPolicy(
        cfg.algo, agent_spec=agent_spec, act_name="drone.target_vel", device="cuda"
    )
    controller = base_env.drone.DEFAULT_CONTROLLER(base_env.drone.dt, 9.81, base_env.drone.params).to(
        base_env.device
    )
    
    def policy(tensordict: TensorDict):
        tensordict = ppo(tensordict)
        relative_state = tensordict[drone_state_key][..., :13].clone()
        if relative_state.dim() > 3:
            relative_state = relative_state.squeeze(2)
        target_vel = tensordict["drone.target_vel"]

        control_target = torch.cat(
            [
                torch.zeros_like(relative_state[..., :3]),
                target_vel,
                torch.zeros_like(target_vel[..., [0]]),
            ],
            dim=-1,
        )
        controller_state = tensordict.get(
            "controller_state", TensorDict({}, relative_state.shape[:2])
        )
        state = torch.cat([
            torch.zeros_like(relative_state[..., :3]),
            relative_state[..., 3:]
        ], dim=-1)
        cmds, controller_state = vmap(vmap(controller))(
            state, control_target, controller_state
        )
        torch.nan_to_num_(cmds, 0.0)
        assert not torch.isnan(cmds).any()
        tensordict["drone.action"] = cmds
        tensordict["controller_state"] = controller_state
        return tensordict

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

    env = TransformedEnv(base_env, Compose(InitTracker(), logger))
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=base_env.num_envs * cfg.algo.train_every,
        total_frames=512000000,
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
        info.update(ppo.train_op(data))

        if i % 2 == 0:
            run.log(info)

        if i % 50 == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            run.log(evaluate())

        pbar.set_postfix({
            "rollout_fps": collector._fps,
            "frames": collector._frames,
        })

    simulation_app.close()


if __name__ == "__main__":
    main()
