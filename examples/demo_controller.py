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

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )
    n = 3

    firefly = Firefly(cfg=RobotCfg())

    translations = torch.zeros(n, 3)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 0.5
    firefly.spawn(translations=translations)

    scene_utils.design_scene()

    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(640, 480),
        data_types=["rgb"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    camera = Camera(
        camera_cfg,
        "/World/envs/env_0/Firefly_0/base_link",
        translation=(-2, -1.4, 0.8),
        target=(0.0, 0.0, 0.0),
    )

    sim.reset()
    firefly.initialize()

    init_poses = firefly.get_env_poses()
    init_vels = firefly.get_velocities()
    controller = LeePositionController(
        dt=sim.get_physics_dt(), g=9.81, uav_params=firefly.params
    ).to(sim.device)

    controller_state = TensorDict({}, n, device=sim.device)
    control_target = torch.zeros(n, 7, device=sim.device)
    control_target[:, 0] = torch.arange(n, device=sim.device).flip(-1) * 0.5
    control_target[:, 1] = torch.arange(n, device=sim.device)
    control_target[:, 2] = 1.0 + torch.arange(n, device=sim.device) * 0.5
    control_target[:, -1] = (torch.pi / 6) * torch.arange(n, device=sim.device)
    action = firefly.action_spec.zero((n,))

    step = 0
    frames = []
    try:
        while simulation_app.is_running():
            if sim.is_stopped() or len(frames) >= 1000:
                break
            if not sim.is_playing():
                continue
            root_state = firefly.get_state()[..., :13].squeeze(0)
            action, controller_state = functorch.vmap(controller)(
                root_state, control_target, controller_state
            )
            firefly.apply_action(action)
            sim.step()

            if step % 2 == 0 and len(frames) < 1000:
                frame = camera().clone()
                frames.append(frame)
                print(step, frame)

            step += 1
            if step >= 1000:
                firefly.set_env_poses(*init_poses)
                firefly.set_velocities(init_vels)
                controller_state.zero_()
                step = 0
                sim.step()
    except KeyboardInterrupt:
        pass
    from torchvision.io import write_video

    for k, v in torch.stack(frames).items():
        write_video(f"{k}.mp4", v.cpu()[..., :3], fps=50)
    simulation_app.close()


if __name__ == "__main__":
    main()
