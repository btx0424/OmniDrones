import torch

import hydra
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app


@hydra.main(version_base=None, config_path=".", config_name="demo")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.controllers import LeePositionController
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.utils.torch import euler_to_quaternion
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

    drone_cls = MultirotorBase.REGISTRY[cfg.drone_model]
    drone = drone_cls()

    translations = torch.zeros(n, 3)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 0.5
    drone.spawn(translations=translations)

    scene_utils.design_scene()

    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(320, 240),
        data_types=["rgb"],
    )
    # cameras used as sensors
    camera_sensor = Camera(camera_cfg)
    camera_sensor.spawn([
        f"/World/envs/env_0/{drone.name}_{i}/base_link/Camera" 
        for i in range(n)
    ])
    # camera for visualization
    camera_vis = Camera(dataclasses.replace(camera_cfg, resolution=(960, 720)))

    sim.reset()
    camera_sensor.initialize(f"/World/envs/env_0/{drone.name}_*/base_link/Camera")
    camera_vis.initialize("/OmniverseKit_Persp")
    drone.initialize()

    # let's fly a circular trajectory
    radius = 1.5
    omega = 1.
    phase = torch.linspace(0, 2, n+1, device=sim.device)[:n]

    def ref(t):
        _t = phase * torch.pi + t * omega
        pos = torch.stack([
            torch.cos(_t) * radius,
            torch.sin(_t) * radius,
            torch.ones(n, device=sim.device) * 1.5
        ], dim=-1)
        vel_xy = torch.stack([
            -torch.sin(_t) * radius * omega,
            torch.cos(_t) * radius * omega,
        ], dim=-1)
        yaw = torch.atan2(vel_xy[:, 1], vel_xy[:, 0])
        return pos, yaw

    init_rpy = torch.zeros(n, 3, device=sim.device)
    init_pos, init_rpy[:, 2] = ref(torch.tensor(0.0).to(sim.device))
    init_rot = euler_to_quaternion(init_rpy)
    init_vels = torch.zeros(n, 6, device=sim.device)

    # create a position controller
    # note: the controller is state-less (but holds its parameters)
    controller = LeePositionController(g=9.81, uav_params=drone.params).to(sim.device)

    def reset():
        drone._reset_idx(torch.tensor([0]))
        drone.set_world_poses(init_pos, init_rot)
        drone.set_velocities(init_vels)
        # flush the buffer so that the next getter invocation 
        # returns up-to-date values
        sim._physics_sim_view.flush() 
    
    reset()
    drone_state = drone.get_state()[..., :13].squeeze(0)

    frames_sensor = []
    frames_vis = []
    from tqdm import tqdm
    for i in tqdm(range(1000)):
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue
        ref_pos, ref_yaw = ref((i % 1000)*cfg.sim.dt)
        action = controller(drone_state, target_pos=ref_pos, target_yaw=ref_yaw)
        drone.apply_action(action)
        sim.step(render=True)

        if i % 2 ==  0:
            frames_sensor.append(camera_sensor.get_images().cpu())
            frames_vis.append(camera_vis.get_images().cpu())

        if i % 1000 == 0:
            reset()
        drone_state = drone.get_state()[..., :13].squeeze(0)

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
