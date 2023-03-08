import torch
import hydra
import os
import time
import wandb
import logging

from tqdm import tqdm
from omegaconf import OmegaConf
from setproctitle import setproctitle
from functorch import vmap
from tensordict import TensorDict
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs.transforms import TransformedEnv

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.learning.collectors import SyncDataCollector
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.envs.transforms import LogOnEpisode

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs import IsaacEnv
    from omni_drones.learning.mappo import MAPPOPolicy
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg
    
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    env = env_class(cfg, headless=cfg.headless)

    agent_spec = env.agent_spec["drone"]
    agent_spec.action_spec = UnboundedContinuousTensorSpec(3, device=env.device)
    ppo = MAPPOPolicy(cfg.algo, agent_spec=agent_spec, act_name="drone.target_vel", device="cuda")
    controller = env.drone.default_controller(
        env.drone.dt, 9.81, env.drone.params
    ).to(env.device)

    def policy(tensordict: TensorDict):
        tensordict = ppo(tensordict)
        relative_state = tensordict["drone.obs"][..., :13].clone()
        target_vel = tensordict["drone.target_vel"]
        
        control_target = torch.cat([
            relative_state[..., :3], 
            target_vel, 
            torch.zeros_like(target_vel[..., [0]])
        ], dim=-1)
        controller_state = tensordict.get("controller_state", TensorDict({}, relative_state.shape[:2]))
        
        cmds, controller_state = vmap(vmap(controller))(relative_state, control_target, controller_state)
        torch.nan_to_num_(cmds, 0.)
        assert not torch.isnan(cmds).any()
        tensordict["drone.action"] = cmds
        tensordict["controller_state"] = controller_state 
        return tensordict

    logger = LogOnEpisode(
        cfg.env.num_envs,
        in_keys=["return", "progress"],
        log_keys=["train/return", "train/ep_length"],
        logger_func=run.log
    )
    
    collector = SyncDataCollector(
        env,
        policy,
        callback=logger, 
        split_trajs=False,
        frames_per_batch=env.num_envs * cfg.algo.train_every,
        device=cfg.sim.device, 
        return_same_td=True,
    )
    
    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(640, 480),
        data_types=["rgb"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1.0e5)
        ),
    )
    camera = Camera(
        camera_cfg, 
        "/World",
        translation=(4., 3., 2.), target=(0., 0., 1)
    )

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
            torch.stack(frames).permute(0, 3, 1, 2),
            fps=1/cfg.sim.dt,
            format="mp4"
        )
        return info

    pbar = tqdm(collector)
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames}
        info.update(ppo.train_op(data))

        if i % 10 == 0:
            run.log(info)
        
        if i % 100 == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            run.log(evaluate())

        pbar.set_postfix({
            "rollout_fps": collector._fps,
            "frames": collector._frames,
            "episodes": collector._episodes,
        })
        
    simulation_app.close()

if __name__ == "__main__":
    main()