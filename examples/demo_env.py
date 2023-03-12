import torch
import hydra
import os
import time
from tensordict import TensorDict
from tqdm import tqdm
from functorch import vmap
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs import Hover
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg

    env = Hover(cfg, headless=cfg.headless)
    controller = env.drone.DEFAULT_CONTROLLER(
        env.drone.dt, 9.81, env.drone.params
    ).to(env.device)

    def policy(tensordict: TensorDict):
        state = tensordict["drone.obs"]
        controller_state = tensordict.get("controller_state", TensorDict({}, state.shape[:2]))
        relative_state = state[..., :13].clone()
        target_pos, quat, linvel, angvel = torch.split(relative_state, [3, 4, 3, 3], dim=-1)
        control_target = torch.cat([
            target_pos, torch.zeros_like(linvel), torch.zeros_like(target_pos[..., [0]])], dim=-1)
        relative_state[..., :3] = 0.
        cmds, controller_state = vmap(vmap(controller))(relative_state, control_target, controller_state)
        tensordict["drone.action"] = cmds
        tensordict["controller_state"] = controller_state
        return tensordict
    
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
        translation=(2.5, 2, 1.5), target=(0., 0., 0.75)
    )
    
    frames = []
    def record_frame(*args, **kwargs):
        frame = camera().clone().cpu()
        frames.append(frame)

    env.enable_render = True
    env.rollout(
        max_steps=1000,
        policy=policy,
        callback=record_frame,
    )

    from torchvision.io import write_video
    for k, v in torch.stack(frames).items():
        write_video(f"demo_env_{k}.mp4", v.cpu()[..., :3], fps=50)
        
    simulation_app.close()

if __name__ == "__main__":
    main()