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
    from omni_drones.utils.torch import euler_to_quaternion, quaternion_to_euler
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
        drone_model_cfg.name, "RateController", cfg.sim.device
    )

    translations = torch.zeros(n, 3)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 0.5
    drone.spawn(translations=translations)

    scene_utils.design_scene()

    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(320, 240),
        data_types=["rgb", "distance_to_camera"],
    )
    # cameras used as sensors
    camera_sensor = Camera(camera_cfg)
    camera_sensor.spawn([
        f"/World/envs/env_0/{drone.name}_{i}/base_link/Camera"
        for i in range(n)
    ])
    # camera for visualization
    # here we reuse the viewport camera, i.e., "/OmniverseKit_Persp"
    camera_vis = Camera(dataclasses.replace(camera_cfg, resolution=(960, 720)))

    sim.reset()
    camera_sensor.initialize(f"/World/envs/env_0/{drone.name}_*/base_link/Camera")
    camera_vis.initialize("/OmniverseKit_Persp")
    drone.initialize()

    init_pos, init_rot = drone.get_world_poses(True)
    init_vels = torch.zeros(n, 6, device=sim.device)
    target_rate = torch.zeros(n, 3, device=sim.device)
    target_rate[:, 2] = torch.pi
    target_height = 1.5 + 0.5 * torch.arange(n, device=sim.device).float()

    def reset():
        drone._reset_idx(torch.tensor([0]))
        drone.set_world_poses(init_pos, init_rot)
        drone.set_velocities(init_vels)

    pos_gain = 4.
    vel_gain = 2.

    def height_control(
        drone,
        target_height = 1.5
    ):
        drone_state = drone.get_state()[..., :13]
        height = drone.pos[..., 2]
        velocity_z = drone.vel[..., 2]
        pos_error = target_height - height
        target_acc = (
            pos_gain * pos_error
            + vel_gain * -velocity_z
            + 9.81
        )
        target_thrust = (target_acc.unsqueeze(-1) * drone.MASS_0)
        cmd = controller(
            drone_state.squeeze(0),
            target_rate=target_rate,
            target_thrust=target_thrust
        )
        return cmd

    reset()

    frames_sensor = []
    frames_vis = []
    from tqdm import tqdm
    for i in tqdm(range(300)):
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue
        action = height_control(drone, target_height)
        drone.apply_action(action)
        sim.step(render=True)

        if i % 2 ==  0:
            frames_sensor.append(camera_sensor.get_images().cpu())
            frames_vis.append(camera_vis.get_images().cpu())

        if i % 300 == 0:
            reset()

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
