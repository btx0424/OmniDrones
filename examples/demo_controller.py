import os

from typing import Dict, Optional
import functorch, torch

import hydra
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app
from tensordict import TensorDict


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni.isaac.core.utils.prims as prim_utils
    import omni_drones.utils.kit as kit_utils
    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.controllers import LeePositionController
    from omni_drones.robots import RobotCfg
    from omni_drones.robots.drone import Crazyflie, Firefly, Hummingbird, MultirotorBase
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg
    from omni_drones.sensors._camera import Camera as OldCam

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )
    n = 4

    firefly = Firefly(cfg=RobotCfg())

    translations = torch.zeros(n, 3)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 0.5
    firefly.spawn(translations=translations)

    scene_utils.design_scene()

    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(640, 480),
        data_types=["rgb", "distance_to_camera"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    camera = Camera(camera_cfg)
    camera.spawn(
        [f"/World/envs/env_0/Firefly_{i}/base_link/Camera" for i in range(n)],
    )

    sim.reset()
    # camera.initialize()
    firefly.initialize()

    init_poses = firefly.get_world_poses(clone=True)
    init_vels = firefly.get_velocities(clone=True)
    controller = LeePositionController(
        dt=sim.get_physics_dt(), g=9.81, uav_params=firefly.params
    ).to(sim.device)

    controller_state = TensorDict({}, n, device=sim.device)
    control_target = torch.zeros(n, 7, device=sim.device)
    control_target[:, 0] = torch.arange(n, device=sim.device).flip(-1) * 0.5
    control_target[:, 1] = torch.arange(n, device=sim.device)
    control_target[:, 2] = 1.0 + torch.arange(n, device=sim.device) * 0.2
    # control_target[:, -1] = (torch.pi / 10) * torch.arange(n, device=sim.device)
    action = firefly.action_spec.zero((n,))

    frames = []
    from tqdm import tqdm
    t = tqdm(range(2000))
    for i in t:
        if sim.is_stopped() or len(frames) >= 1000:
            break
        if not sim.is_playing():
            continue
        root_state = firefly.get_state(env=False)[..., :13].squeeze(0)
        action, controller_state = functorch.vmap(controller)(
            root_state, control_target, controller_state
        )
        firefly.apply_action(action)
        sim.step(i % 2 == 0)

        if i % 2 == 0 and len(frames) < 1000:
            frame = camera.get_images()
            frames.append(frame.cpu())

        if i % 1000 == 0:
            firefly.set_world_poses(*init_poses)
            firefly.set_velocities(init_vels)
            controller_state.zero_()
            sim.step()
        
    from torchvision.io import write_video

    for k, v in torch.stack(frames).cpu().items():
        torch.save(v, f"{k}.pth")
        for i, vv in enumerate(v.unbind(1)):
            if vv.shape[-1] == 4:
                write_video(f"{k}_{i}.mp4", vv[..., :3], fps=50)
            elif vv.shape[-1] == 1:
                vv = -torch.nan_to_num(vv, 0).expand(*vv.shape[:-1], 3)
                write_video(f"{k}_{i}.mp4", vv[..., :3], fps=50)
        
    simulation_app.close()


if __name__ == "__main__":
    main()
