import os

from typing import Dict, Optional
import torch

import hydra
from omegaconf import OmegaConf
from omni_drones import init_simulation_app
from tensordict import TensorDict


@hydra.main(version_base=None, config_path=".", config_name="demo")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg
    import dataclasses

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=cfg.sim.dt,
        rendering_dt=cfg.sim.dt,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )
    n = 4

    drone_model_cfg = cfg.drone_model
    drone, controller = MultirotorBase.make(
        drone_model_cfg.name, drone_model_cfg.controller, cfg.sim.device
    )

    translations = torch.tensor([
        [0, -1, 1.5],
        [0, 0., 1.5],
        [0, 1., 1.5],
        [0., 2., 2.5]
    ])
    drone.spawn(translations=translations)
    scene_utils.design_scene()

    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(960, 720),
        data_types=["rgb"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    # cameras used as sensors
    camera_sensor = Camera(camera_cfg)
    camera_sensor.spawn(
        ["/World/Camera"],
        translations=[(8, 2., 3.)],
        targets=[(0., 0., 1.75)]
    )
    # camera for visualization
    # here we reuse the viewport camera, i.e., "/OmniverseKit_Persp"
    camera_vis = Camera(dataclasses.replace(camera_cfg, resolution=(960, 720)))


    sim.reset()
    camera_sensor.initialize("/World/Camera")
    camera_vis.initialize("/OmniverseKit_Persp")
    drone.initialize()

    target_pos = torch.zeros(n, 3, device=sim.device)
    target_pos[:, 0] = 0
    target_pos[:, 1] = translations[:, 1]
    target_pos[:, 2] = translations[:, 2]
    action = drone.action_spec.zero((n,))

    frames_sensor = []
    frames_vis = []
    from tqdm import tqdm
    t = tqdm(range(cfg.steps))
    for i in t:
        if sim.is_stopped():
            break
        if not sim.is_playing():
            continue
        root_state = drone.get_state()[..., :13].squeeze(0)
        distance = torch.norm(root_state[-1, :2] - target_pos[-1, :2])
        if distance < 0.05:
            target_pos[-1, 1] = -target_pos[-1, 1]
        action = controller.compute(root_state, target_pos=target_pos)
        drone.apply_action(action)
        sim.step(i % 2 == 0)

        if i % 2 == 0:
            frames_sensor.append(camera_sensor.get_images().cpu())
            frames_vis.append(camera_vis.get_images().cpu())

    # write videos
    from torchvision.io import write_video

    for image_type, arrays in torch.stack(frames_sensor).items():
        print(f"Writing {image_type} of shape {arrays.shape}.")
        for drone_id, arrays_drone in enumerate(arrays.unbind(1)):
            if image_type == "rgb":
                arrays_drone = arrays_drone.permute(0, 2, 3, 1)[..., :3]
                write_video(f"demo_rgb_{drone_id}.mp4", arrays_drone, fps=1/cfg.sim.dt)
            elif image_type == "distance_to_camera":
                arrays_drone = -torch.nan_to_num(arrays_drone, 0).permute(0, 2, 3, 1)
                arrays_drone = arrays_drone.expand(*arrays_drone.shape[:-1], 3)
                write_video(f"demo_depth_{drone_id}.mp4", arrays_drone, fps=0.5/cfg.sim.dt)

    for image_type, arrays in torch.stack(frames_vis).items():
        print(f"Writing {image_type} of shape {arrays.shape}.")
        for _, arrays_drone in enumerate(arrays.unbind(1)):
            if image_type == "rgb":
                arrays_drone = arrays_drone.permute(0, 2, 3, 1)[..., :3]
                write_video(f"demo_rgb.mp4", arrays_drone, fps=1/cfg.sim.dt)
            elif image_type == "distance_to_camera":
                arrays_drone = -torch.nan_to_num(arrays_drone, 0).permute(0, 2, 3, 1)
                arrays_drone = arrays_drone.expand(*arrays_drone.shape[:-1], 3)
                write_video(f"demo_depth.mp4", arrays_drone, fps=0.5/cfg.sim.dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
