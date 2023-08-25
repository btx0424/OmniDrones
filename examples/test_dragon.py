import torch

import hydra
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app
from torchrl.data import BoundedTensorSpec
from tensordict.nn import make_functional


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

    from omni_drones.utils.torch import euler_to_quaternion, quat_axis
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg
    import dataclasses
    from dataclasses import dataclass, field, MISSING, fields, asdict
    
    @dataclass
    class RotorConfig:
        directions: torch.Tensor = MISSING
        max_rotation_velocities: torch.Tensor = MISSING
        force_constants: torch.Tensor = MISSING
        moment_constants: torch.Tensor = MISSING

        num_rotors: int = field(init=False)

        def __post_init__(self):
            for f in fields(self):
                if f.type == torch.Tensor:
                    setattr(self, f.name, torch.as_tensor(getattr(self, f.name)))
            self.num_rotors = len(self.directions)
            print(self)



    @dataclass
    class DragonCfg(RobotCfg):

        rotor_cfg: RotorConfig = RotorConfig(
            directions=[1, -1, 1, -1, 1, -1, 1, -1],
            force_constants=torch.ones(8) * 7.2e-6,
            moment_constants=torch.ones(8) * 1.08e-7,
            max_rotation_velocities=torch.ones(8) * 1700
        )


    class Dragon(MultirotorBase):

        usd_path: str = ASSET_PATH + "/usd/dragon-4.usd"

        def __init__(self, name: str = "dragon", cfg: DragonCfg = DragonCfg(), is_articulation: bool = True) -> None:
            super(MultirotorBase, self).__init__(name, cfg, is_articulation)
            self.num_rotors = self.cfg.rotor_cfg.num_rotors
            self.num_links = 4
            self.action_split = [self.cfg.rotor_cfg.num_rotors, self.num_links * 2, (self.num_links-1) * 2]
            action_dim = sum(self.action_split)
            self.action_spec = BoundedTensorSpec(-1, 1, action_dim, device=self.device)

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
                # shape=self.shape,
            )
            self.base_link.initialize()
            self.rotor_joint_indices = [i for i, name in enumerate(self._view.dof_names) if name.startswith("rotor_")]
            self.link_joint_indices = [i for i, name in enumerate(self._view.dof_names) if name.startswith("joint_")]
            # self.link_joint_limits = self._view.get_dof_limits().clone()[..., self.link_joint_indices]
            self.gimbal_joint_indices = [i for i, name in enumerate(self._view.dof_names) if name.startswith("D6Joint")]
            self.masses = self._view.get_body_masses()

            self.rotors = RotorGroup(asdict(self.cfg.rotor_cfg), self.dt).to(self.device)
            rotor_params = make_functional(self.rotors)
            self.rotor_params = rotor_params.expand(self.shape).clone()
            self.throttle = self.rotor_params["throttle"]

            self.thrusts = torch.zeros(*self.shape, self.cfg.rotor_cfg.num_rotors, 3, device=self.device)
            self.torques = torch.zeros(*self.shape, 4, 3, device=self.device)
        
        def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
            rotor_cmds, gimbal_cmds, link_cmds = actions.split(self.action_split, dim=-1)
            rotor_cmds = rotor_cmds.expand(*self.shape, self.cfg.rotor_cfg.num_rotors)
            rotor_pos, rotor_rot = self.rotors_view.get_world_poses()
            torque_axis = quat_axis(rotor_rot, axis=2)
            thrusts, moments = self.rotors(rotor_cmds, params=self.rotor_params)

            self.thrusts[..., 2] = thrusts
            self.torques[:] = (moments.unsqueeze(-1) * torque_axis).sum(-2)

            self.rotors_view.apply_forces_and_torques_at_pos(
                self.thrusts.reshape(-1, 3), 
                is_global=False
            )
            self.base_link.apply_forces_and_torques_at_pos(
                torques = self.torques.reshape(-1, 3), 
                is_global=True
            )
            
            self._view.set_joint_velocity_targets(
                (gimbal_cmds * torch.pi).reshape(-1, 8), joint_indices=self.gimbal_joint_indices
            )
            # (link_cmds * (self.link_joint_limits[..., 1] - self.link_joint_limits[..., 0])
            #      + self.link_joint_limits[..., 0])
            self._view.set_joint_position_targets(
                (link_cmds * torch.pi/2).reshape(-1, 6), 
                joint_indices=self.link_joint_indices
            )
            return self.throttle.sum(-1)
    

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

    scene_utils.design_scene()

    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(960, 720),
        data_types=["rgb", "distance_to_camera"],
    )
    # # camera for visualization
    camera_vis = Camera(camera_cfg)

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
    action = drone.action_spec.rand()
    action[..., :8] = -1.
    from tqdm import tqdm
    for i in tqdm(range(1000)):
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue

        drone.apply_action(action)
        sim.step(render=True)

        # drone_state = drone.get_state()[..., :13].squeeze(0)

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
