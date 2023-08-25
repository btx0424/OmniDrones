import torch

import hydra
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.actuators.rotor_group import RotorGroup

    from omni_drones.robots import ASSET_PATH, RobotCfg
    from omni_drones.views import RigidPrimView

    from omni_drones.utils.torch import euler_to_quaternion
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg
    import dataclasses
    from dataclasses import dataclass, field, MISSING, fields, asdict
    
    @dataclass
    class RotorConfig:
        directions: torch.Tensor = MISSING
        max_rotation_vels: torch.Tensor = MISSING
        force_constants: torch.Tensor = MISSING
        moment_constants: torch.Tensor = MISSING

        def __post_init__(self):
            for f in fields(self):
                if f.type == torch.Tensor:
                    setattr(self, f.name, torch.as_tensor(getattr(self, f.name)))
            print(self)



    @dataclass
    class DragonCfg(RobotCfg):

        rotor_cfg: RotorConfig = RotorConfig(
            directions=[1, -1, 1, -1, 1, -1, 1, -1],
            force_constants=torch.ones(8) * 7.2e-6,
            moment_constants=torch.ones(8) * 1.08e-7,
            max_rotation_vels=torch.ones(8) * 1700
        )


    class Dragon(MultirotorBase):

        usd_path: str = ASSET_PATH + "/usd/dragon-4.usd"

        def __init__(self, name: str = "dragon", cfg: DragonCfg = DragonCfg(), is_articulation: bool = True) -> None:
            super(MultirotorBase, self).__init__(name, cfg, is_articulation)
        
        def initialize(self, prim_paths_expr: str = None, track_contact_forces: bool = False):
            super(MultirotorBase, self).initialize(prim_paths_expr)

            self.rotors_view = RigidPrimView(
                prim_paths_expr=f"{self.prim_paths_expr}/link_*/rotor_*",
                name="rotors",
                shape=(*self.shape, -1)
            )
            self.rotors_view.initialize()
            self.base_link = RigidPrimView(
                prim_paths_expr=f"{self.prim_paths_expr}/link_*/base_link",
                name="base_link",
                # track_contact_forces=track_contact_forces,
                shape=self.shape,
            )
            self.base_link.initialize()
            self.rotor_joint_indices = [i for i, name in enumerate(self._view.dof_names) if name.startswith("rotor_")]
            self.link_joint_indices = [i for i, name in enumerate(self._view.dof_names) if name.startswith("joint_")]
            self.gimbal_joint_indices = [i for i, name in enumerate(self._view.dof_names) if name.startswith("D6joint")]
            
            self.rotors = RotorGroup(asdict(self.cfg.rotor_cfg), self.dt)
            print()
    

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )
    drone = Dragon()

    translations = torch.zeros(1, 3)
    translations[:, 2] = 1.5
    drone.spawn(translations=translations)

    # scene_utils.design_scene()

    # camera_cfg = PinholeCameraCfg(
    #     sensor_tick=0,
    #     resolution=(320, 240),
    #     data_types=["rgb", "distance_to_camera"],
    # )
    # # cameras used as sensors
    # camera_sensor = Camera(camera_cfg)
    # camera_sensor.spawn([
    #     f"/World/envs/env_0/{drone.name}_{i}/base_link/Camera" 
    #     for i in range(n)
    # ])
    # # camera for visualization
    # camera_vis = Camera(dataclasses.replace(camera_cfg, resolution=(960, 720)))

    sim.reset()
    # camera_sensor.initialize(f"/World/envs/env_0/{drone.name}_*/base_link/Camera")
    # camera_vis.initialize("/OmniverseKit_Persp")
    drone.initialize()

    # # let's fly a circular trajectory
    # radius = 1.5
    # omega = 1.
    # phase = torch.linspace(0, 2, n+1, device=sim.device)[:n]

    # def ref(t):
    #     _t = phase * torch.pi + t * omega
    #     pos = torch.stack([
    #         torch.cos(_t) * radius,
    #         torch.sin(_t) * radius,
    #         torch.ones(n, device=sim.device) * 1.5
    #     ], dim=-1)
    #     vel_xy = torch.stack([
    #         -torch.sin(_t) * radius * omega,
    #         torch.cos(_t) * radius * omega,
    #     ], dim=-1)
    #     yaw = torch.atan2(vel_xy[:, 1], vel_xy[:, 0])
    #     return pos, yaw

    # init_rpy = torch.zeros(n, 3, device=sim.device)
    # init_pos, init_rpy[:, 2] = ref(torch.tensor(0.0).to(sim.device))
    # init_rot = euler_to_quaternion(init_rpy)
    # init_vels = torch.zeros(n, 6, device=sim.device)

    # # create a position controller
    # # note: the controller is state-less (but holds its parameters)
    # controller = LeePositionController(g=9.81, uav_params=drone.params).to(sim.device)

    # def reset():
    #     drone._reset_idx(torch.tensor([0]))
    #     drone.set_world_poses(init_pos, init_rot)
    #     drone.set_velocities(init_vels)
    #     # flush the buffer so that the next getter invocation 
    #     # returns up-to-date values
    #     sim._physics_sim_view.flush() 
    
    # reset()
    # drone_state = drone.get_state()[..., :13].squeeze(0)

    # frames_sensor = []
    # frames_vis = []
    # from tqdm import tqdm
    # for i in tqdm(range(1000)):
    #     if sim.is_stopped():
    #         break
    #     if not sim.is_playing():
    #         sim.render()
    #         continue
    #     ref_pos, ref_yaw = ref((i % 1000)*cfg.sim.dt)
    #     action = controller(drone_state, target_pos=ref_pos, target_yaw=ref_yaw)
    #     drone.apply_action(action)
    #     sim.step(render=True)

    #     if i % 2 ==  0:
    #         frames_sensor.append(camera_sensor.get_images().cpu())
    #         frames_vis.append(camera_vis.get_images().cpu())

    #     if i % 1000 == 0:
    #         reset()
    #     drone_state = drone.get_state()[..., :13].squeeze(0)

    # from torchvision.io import write_video

    # for image_type, arrays in torch.stack(frames_sensor).items():
    #     print(f"Writing {image_type} of shape {arrays.shape}.")
    #     for drone_id, arrays_drone in enumerate(arrays.unbind(1)):
    #         if image_type == "rgb":
    #             arrays_drone = arrays_drone.permute(0, 2, 3, 1)[..., :3]
    #             write_video(f"rgb_{drone_id}.mp4", arrays_drone, fps=1/cfg.sim.dt)
    #         elif image_type == "distance_to_camera":
    #             arrays_drone = -torch.nan_to_num(arrays_drone, 0).permute(0, 2, 3, 1)
    #             arrays_drone = arrays_drone.expand(*arrays_drone.shape[:-1], 3)
    #             write_video(f"depth_{drone_id}.mp4", arrays_drone, fps=1/cfg.sim.dt)

    # for image_type, arrays in torch.stack(frames_vis).items():
    #     print(f"Writing {image_type} of shape {arrays.shape}.")
    #     for _, arrays_drone in enumerate(arrays.unbind(1)):
    #         if image_type == "rgb":
    #             arrays_drone = arrays_drone.permute(0, 2, 3, 1)[..., :3]
    #             write_video(f"rgb.mp4", arrays_drone, fps=1/cfg.sim.dt)
    #         elif image_type == "distance_to_camera":
    #             arrays_drone = -torch.nan_to_num(arrays_drone, 0).permute(0, 2, 3, 1)
    #             arrays_drone = arrays_drone.expand(*arrays_drone.shape[:-1], 3)
    #             write_video(f"depth.mp4", arrays_drone, fps=1/cfg.sim.dt)

    # simulation_app.close()


if __name__ == "__main__":
    main()
