import os

from typing import Dict, Optional
import torch
from functorch import vmap

import hydra
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app
from tensordict import TensorDict


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.controllers import LeePositionController
    from omni_drones.robots.drone import MultirotorBase
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
        resolution=(320, 240),
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
        [f"/World/envs/env_0/{drone.name}_{i}/base_link/Camera" for i in range(n)],
    )

    sim.reset()
    camera.initialize(f"/World/envs/env_0/{drone.name}_*/base_link/Camera")
    drone.initialize()

    # let's fly a circular trajectory
    radius = 1.5
    omega = 1.
    phase = torch.linspace(0, 2, n+1)[:n]
    def ref_pos(t):
        _t = phase * torch.pi + t * omega
        pos = torch.stack([
            torch.cos(_t) * radius,
            torch.sin(_t) * radius,
            torch.ones(n) * 1.5
        ], dim=-1)
        return pos
    init_poses = (
        ref_pos(0).to(sim.device), 
        torch.tensor([1., 0., 0.,0.], device=sim.device).repeat(n, 1)
    )
    init_vels = torch.zeros(n, 6, device=sim.device)

    # create a position controller
    # note: the controller is state-less (but holds its parameters)
    controller = LeePositionController(
        dt=sim.get_physics_dt(), g=9.81, uav_params=drone.params
    ).to(sim.device)
    controller_fun = vmap(controller)
    control_target = torch.zeros(n, 7, device=sim.device)

    def reset():
        drone._reset_idx(torch.tensor([0]))
        drone.set_world_poses(*init_poses)
        drone.set_velocities(init_vels)
        # flush the buffer so that the next getter invocation 
        # returns up-to-date values
        sim._physics_sim_view.flush() 
    
    reset()
    drone_state = drone.get_state()[..., :13].squeeze(0)

    frames = []
    from tqdm import tqdm
    for i in tqdm(range(2000)):
        if sim.is_stopped() or len(frames) >= 1000:
            break
        if not sim.is_playing():
            sim.render(not cfg.headless)
            continue
        control_target[:, :3] = ref_pos((i % 1000)*cfg.sim.dt)
        action, _ = controller_fun(drone_state, control_target, {})
        drone.apply_action(action)
        sim.step(not cfg.headless)

        if i % 2 == 0 and len(frames) < 1000:
            frame = camera.get_images()
            frames.append(frame.cpu())

        if i % 1000 == 0:
            reset()
        drone_state = drone.get_state()[..., :13].squeeze(0)
        
    from torchvision.io import write_video
    video_arrays = torch.stack(frames)

    for image_type, arrays in video_arrays.items():
        for drone_id, vv in enumerate(arrays.unbind(1)):
            if image_type == "rgb":
                write_video(f"rgb_{drone_id}.mp4", vv[..., :3], fps=1/cfg.sim.dt)
            elif image_type == "distance_to_camera":
                vv = -torch.nan_to_num(vv, 0).expand(*vv.shape[:-1], 3)
                write_video(f"depth_{drone_id}.mp4", vv[..., :3], fps=1/cfg.sim.dt)
        
    simulation_app.close()


if __name__ == "__main__":
    main()
