import torch
from dataclasses import dataclass, field, MISSING, fields, asdict

from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec
from tensordict.nn import make_functional

from omni_drones.robots import ASSET_PATH, RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import quat_axis

from collections import defaultdict

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

    force_sensor: bool = False
    rotor_cfg: RotorConfig = RotorConfig(
        directions=[1, -1, 1, -1, 1, -1, 1, -1],
        force_constants=torch.ones(8) * 7.2e-6,
        moment_constants=torch.ones(8) * 1.08e-7,
        max_rotation_velocities=torch.ones(8) * 800
    )


class Dragon(MultirotorBase):

    usd_path: str = ASSET_PATH + "/usd/dragon-4-1.usd"
    cfg_cls = DragonCfg

    def __init__(self, name: str = "dragon", cfg: DragonCfg = DragonCfg(), is_articulation: bool = True) -> None:
        super(MultirotorBase, self).__init__(name, cfg, is_articulation)
        self.num_rotors = self.cfg.rotor_cfg.num_rotors
        self.num_links = 4
        self.action_split = [self.cfg.rotor_cfg.num_rotors, self.num_links * 2, (self.num_links-1) * 2]
        action_dim = sum(self.action_split)
        self.action_spec = BoundedTensorSpec(-1, 1, action_dim, device=self.device)
        observation_dim = (
            self.num_links * (
                13 + 3 + 3
                + (2 + 2) # gimbal module
            )
            # + (self.num_links-1) # link joint
        )
        self.state_spec = UnboundedContinuousTensorSpec(observation_dim, device=self.device)
        self.intrinsics_spec = CompositeSpec({
            "KF": UnboundedContinuousTensorSpec(self.num_rotors),
            "KM": UnboundedContinuousTensorSpec(self.num_rotors),
        }).to(self.device)
        self.randomization = defaultdict(dict)

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
            shape=(*self.shape, self.num_links),
        )
        self.base_link.initialize()
        self.pos = torch.zeros(*self.shape, self.num_links, 3, device=self.device)
        self.rot = torch.zeros(*self.shape, self.num_links, 4, device=self.device)
        self.vel = torch.zeros(*self.shape, self.num_links, 6, device=self.device)
        self.acc = torch.zeros(*self.shape, self.num_links, 6, device=self.device)
        
        self.rotor_joint_indices = [i for i, name in enumerate(self._view.dof_names) if name.startswith("rotor_")]
        self.link_joint_indices = [i for i, name in enumerate(self._view.dof_names) if name.startswith("link_joint")]
        # self.link_joint_limits = self._view.get_dof_limits().clone()[..., self.link_joint_indices]
        self.gimbal_joint_indices = [i for i, name in enumerate(self._view.dof_names) if name.startswith("gimbal_")]
        self.joint_gains = self._view.get_gains(clone=True)
        self.body_masses = self._view.get_body_masses(clone=True)
        self.gravity = self.body_masses.sum(-1, keepdim=True) * 9.81

        self.rotors = RotorGroup(asdict(self.cfg.rotor_cfg), self.dt).to(self.device)
        rotor_params = make_functional(self.rotors)
        self.rotor_params = rotor_params.expand(self.shape).clone()
        self.throttle = self.rotor_params["throttle"]
        self.throttle_difference = torch.zeros_like(self.throttle)
        self.KF = self.rotor_params["KF"]
        self.KM = self.rotor_params["KM"]

        self.thrusts = torch.zeros(*self.shape, self.cfg.rotor_cfg.num_rotors, 3, device=self.device)
        self.torques = torch.zeros(*self.shape, 4, 3, device=self.device)

        self.intrinsics = self.intrinsics_spec.expand(self.shape).zero()

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds, gimbal_cmds, link_cmds = actions.split(self.action_split, dim=-1)
        rotor_cmds = rotor_cmds.expand(*self.shape, self.cfg.rotor_cfg.num_rotors)
        rotor_pos, rotor_rot = self.rotors_view.get_world_poses()
        torque_axis = quat_axis(rotor_rot, axis=2)
        thrusts, moments = self.rotors(rotor_cmds, params=self.rotor_params)

        self.thrusts[..., 2] = thrusts
        self.torques[:] = (
            (moments.unsqueeze(-1) * torque_axis)
            .reshape(*self.shape, self.num_links, 2, 3)
            .sum(-2)
        )

        self.rotors_view.apply_forces_and_torques_at_pos(
            self.thrusts.reshape(-1, 3), 
            is_global=False
        )
        self.base_link.apply_forces_and_torques_at_pos(
            torques = self.torques.reshape(-1, 3), 
            is_global=True
        )
        gimbal_cmds = gimbal_cmds.clamp(-1, 1) * torch.pi / 2
        self._view.set_joint_position_targets(
            gimbal_cmds.reshape(-1, 8), joint_indices=self.gimbal_joint_indices
        )
        # (link_cmds * (self.link_joint_limits[..., 1] - self.link_joint_limits[..., 0])
        #      + self.link_joint_limits[..., 0])
        # link_cmds = link_cmds.clamp(-1, 1) * torch.pi / 2
        # self._view.set_joint_position_targets(
        #     link_cmds.reshape(-1, 6), 
        #     joint_indices=self.link_joint_indices
        # )
        return self.throttle.sum(-1)
    
    def get_state(self, check_nan: bool = False):
        self.pos[:], self.rot[:] = self.base_link.get_world_poses()
        if hasattr(self, "_envs_positions"):
            self.pos.sub_(self._envs_positions.unsqueeze(2))
        joint_pos = self.get_joint_positions(clone=True) / torch.pi
        link_joint_pos = joint_pos[..., self.link_joint_indices]
        gimbal_joint_pos = joint_pos[..., self.gimbal_joint_indices]
        vel = self.base_link.get_velocities(True)
        self.vel[:] = vel
        self.heading = quat_axis(self.rot, axis=0)
        self.up = quat_axis(self.rot, axis=2)
        state = [self.pos, self.rot, self.vel, self.heading, self.up, (self.throttle * 2 - 1).reshape(*self.shape, self.num_links, 2)]
        # state.append(link_joint_pos)
        state.append(gimbal_joint_pos.reshape(*self.shape, self.num_links, 2))
        state = torch.cat(state, dim=-1)
        if check_nan:
            assert not torch.isnan(state).any()
        return state

