import os

from typing import Dict, Optional
import torch
from functorch import vmap

import hydra
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app
from tensordict import TensorDict


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.controllers import AttitudeController
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.utils.torch import euler_to_quaternion, quaternion_to_euler
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )
    n = 4

    drone_model = "Hummingbird"
    drone_cls = MultirotorBase.REGISTRY[drone_model]
    drone = drone_cls()

    translations = torch.zeros(n, 3)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 0.5
    drone.spawn(translations=translations)

    scene_utils.design_scene()

    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(960, 720),
        data_types=["rgb", "distance_to_camera"],
    )
    # camera for visualization
    camera_vis = Camera(camera_cfg)

    sim.reset()
    camera_vis.initialize("/OmniverseKit_Persp")
    drone.initialize()

    init_pos, init_rot = drone.get_world_poses(True)
    init_vels = torch.zeros(n, 6, device=sim.device)
    target_yaw_rate = torch.zeros(n, 1, device=sim.device)
    # target_yaw_rate[:] = torch.pi
    target_pitch = torch.zeros(n, 1, device=sim.device)
    target_pitch[:] = torch.pi/6

    controller = AttitudeController(9.8, uav_params=drone.params).to(sim.device)

    def reset():
        drone._reset_idx(torch.tensor([0]))
        drone.set_world_poses(init_pos, init_rot)
        drone.set_velocities(init_vels)
        # flush the buffer so that the next getter invocation 
        # returns up-to-date values
        sim._physics_sim_view.flush() 
    
    reset()
    drone_state = drone.get_state()[..., :13].squeeze(0)

    frames_vis = []
    
    from tqdm import tqdm
    for i in tqdm(range(500)):
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue
        action = controller(
            drone_state, 
            target_yaw_rate=target_yaw_rate, 
            target_pitch=target_pitch,
            target_thrust=(drone.MASS_0 * 10.)
        )
        drone.apply_action(action)
        sim.step(render=True)

        if i % 2 ==  0:
            frames_vis.append(camera_vis.get_images().cpu())

        if i % 1000 == 0:
            reset()
        drone_state = drone.get_state()[..., :13].squeeze(0)

    from torchvision.io import write_video


    for image_type, arrays in torch.stack(frames_vis).items():
        print(f"Writing {image_type} of shape {arrays.shape}.")
        for _, arrays_drone in enumerate(arrays.unbind(1)):
            if image_type == "rgb":
                arrays_drone = arrays_drone.permute(0, 2, 3, 1)[..., :3]
                write_video(f"rgb.mp4", arrays_drone, fps=1/cfg.sim.dt)
            elif image_type == "distance_to_camera":
                arrays_drone = -torch.nan_to_num(arrays_drone, 0).permute(0, 2, 3, 1)
                arrays_drone = arrays_drone.expand(*arrays_drone.shape[:-1], 3)
                write_video(f"depth.mp4", arrays_drone, fps=1/cfg.sim.dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
