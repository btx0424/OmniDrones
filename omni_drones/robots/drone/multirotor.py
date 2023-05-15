import logging
from typing import Type

import omni.isaac.core.utils.torch as torch_utils
import torch
import yaml
from functorch import vmap
from tensordict.nn import make_functional
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec

from omni_drones.views import RigidPrimView
from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.controllers import LeePositionController

from omni_drones.robots import RobotBase, RobotCfg
from omni_drones.utils.torch import normalize, off_diag

from dataclasses import dataclass

@dataclass
class MultirotorCfg(RobotCfg):
    force_sensor: bool = False

class MultirotorBase(RobotBase):

    param_path: str
    DEFAULT_CONTROLLER: Type = LeePositionController
    cfg_cls = MultirotorCfg

    def __init__(
        self, 
        name: str = None, 
        cfg: MultirotorCfg=None, 
        is_articulation: bool = True
    ) -> None:
        super().__init__(name, cfg, is_articulation)

        with open(self.param_path, "r") as f:
            logging.info(f"Reading {self.name}'s params from {self.param_path}.")
            self.params = yaml.safe_load(f)
        self.mass = self.params["mass"]
        self.num_rotors = self.params["rotor_configuration"]["num_rotors"]

        self.action_spec = BoundedTensorSpec(-1, 1, self.num_rotors, device=self.device)
        if self.cfg.force_sensor:
            self.use_force_sensor = True
            state_dim = 19 + self.num_rotors + 6
        else:
            self.use_force_sensor = False
            state_dim = 19 + self.num_rotors
        self.state_spec = UnboundedContinuousTensorSpec(state_dim, device=self.device)

    def initialize(self, prim_paths_expr: str = None):
        if self.is_articulation:
            super().initialize(prim_paths_expr=prim_paths_expr)
            self.base_link = RigidPrimView(
                prim_paths_expr=f"{self.prim_paths_expr}/base_link",
                name="base_link",
                # track_contact_forces=True,
                shape=self.shape,
            )
            self.base_link.initialize()
            print(self._view.dof_names)
            print(self._view._dof_indices)
        else:
            super().initialize(prim_paths_expr=f"{prim_paths_expr}/base_link")
            self.base_link = self._view
            self.prim_paths_expr = prim_paths_expr

        self.rotors_view = RigidPrimView(
            # prim_paths_expr=f"{self.prim_paths_expr}/rotor_[0-{self.num_rotors-1}]",
            prim_paths_expr=f"{self.prim_paths_expr}/rotor_*",
            name="rotors",
            shape=(*self.shape, self.num_rotors)
        )
        self.rotors_view.initialize()

        self.rotors = RotorGroup(self.params["rotor_configuration"], dt=self.dt).to(
            self.device
        )
        self.rotor_params_and_states = (
            make_functional(self.rotors).expand(self.shape).clone()
        )

        self.MAX_ROT_VEL = self.rotor_params_and_states["MAX_ROT_VEL"]
        self.KF = self.rotor_params_and_states["KF"]
        self.KM = self.rotor_params_and_states["KM"]

        self.max_forces = self.rotor_params_and_states["max_forces"]
        self.throttle = self.rotor_params_and_states["throttle"]
        self.directions = self.rotor_params_and_states["directions"]

        self.forces = torch.zeros(*self.shape, self.num_rotors, 3, device=self.device)
        self.torques = torch.zeros(*self.shape, 3, device=self.device)

        self.pos, self.rot = self.get_world_poses(True)
        self.throttle_difference = torch.zeros(self.throttle.shape[:-1], device=self.device)
        self.heading = torch.zeros(*self.shape, 3, device=self.device)
        self.up = torch.zeros(*self.shape, 3, device=self.device)
        self.vel = torch.zeros(*self.shape, 6, device=self.device)
        self.acc = torch.zeros(*self.shape, 6, device=self.device)
        self.jerk = torch.zeros(*self.shape, 6, device=self.device)
        self.alpha = 0.9

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds = actions.expand(*self.shape, self.num_rotors)
        last_throttle = self.throttle.clone()
        thrusts, moments = vmap(vmap(self.rotors, randomness="different"), randomness="same")(
            rotor_cmds, self.rotor_params_and_states
        )
        self.forces[..., 2] = thrusts
        self.torques[..., 2] = moments.sum(-1)
        # self.articulations.set_joint_velocities(
        #     (self.throttle * self.directions * self.MAX_ROT_VEL).reshape(-1, self.num_rotors)
        # )

        # TODO: global downwash
        if self.n > 1:
            downwash_forces = vmap(self.downwash)(
                self.pos,
                self.pos,
                vmap(torch_utils.quat_rotate)(self.rot, self.forces.sum(-2)),
                kz=0.3
            ).sum(-2).reshape(-1, 3)
        else:
            downwash_forces = None
        torques = vmap(torch_utils.quat_rotate)(self.rot, self.torques).reshape(-1, 3)

        self.rotors_view.apply_forces(self.forces.reshape(-1, 3), is_global=False)
        self.base_link.apply_forces_and_torques_at_pos(
            downwash_forces, 
            torques, 
            is_global=True
        )
        self.throttle_difference[:] = torch.norm(self.throttle - last_throttle, dim=-1)
        return self.throttle.sum(-1)

    def get_state(self, env=True, check_nan: bool=False):
        self.pos[:], self.rot[:] = self.get_world_poses(True)
        if env:
            self.pos[:] = self.pos[:] - RobotBase._envs_positions
        vel = self.get_velocities(True)
        acc = self.acc.lerp((vel - self.vel) / self.dt, self.alpha)
        jerk = self.jerk.lerp((acc - self.acc) / self.dt, self.alpha)
        self.jerk[:] = jerk
        self.acc[:] = acc
        self.vel[:] = vel
        self.heading[:] = vmap(torch_utils.quat_axis)(self.rot, axis=0)
        self.up[:] = vmap(torch_utils.quat_axis)(self.rot, axis=2)
        state = [self.pos, self.rot, self.vel, self.heading, self.up, self.throttle * 2 - 1]
        if self.use_force_sensor:
            self.force_sensor_readings = self.get_force_sensor_forces() 
            state.append(self.force_sensor_readings.flatten(-2)/ self.mass)
        state = torch.cat(state, dim=-1)
        if check_nan:
            assert not torch.isnan(state).any()
        return state

    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids = torch.arange(self.shape[0], device=self.device)
        self.forces[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.throttle[env_ids] = self.rotors.f_inv(1 / self.get_thrust_to_weight_ratio()[env_ids])
        self.throttle_difference[env_ids] = 0.0
        self.vel[env_ids] = 0.
        self.acc[env_ids] = 0.
        self.jerk[env_ids] = 0.
        return env_ids
    
    def get_thrust_to_weight_ratio(self):
        return self.max_forces.sum(-1, keepdim=True) / (self.mass * 9.81)

    def get_linear_smoothness(self):
        return - (
            torch.norm(self.acc[..., :3], dim=-1) 
            + torch.norm(self.jerk[..., :3], dim=-1)
        )
    
    def get_angular_smoothness(self):
        return - (
            torch.sum(self.acc[..., 3:].abs(), dim=-1)
            + torch.sum(self.jerk[..., 3:].abs(), dim=-1)
        )
    
    @staticmethod
    def downwash(
        p0: torch.Tensor, 
        p1: torch.Tensor,
        p1_t: torch.Tensor,
        kr: float=2,
        kz: float=1,
    ):
        """
        A highly simplified downwash effect model. 
        
        References:
        https://arxiv.org/pdf/2207.09645.pdf
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8798116

        """
        z, r = separation(p0, p1, normalize(p1_t))
        z = torch.clip(z, 0)
        v = torch.exp(-0.5 * torch.square(kr * r / z)) / (1 + kz * z)**2
        f = off_diag(v * - p1_t)
        return f

def separation(p0, p1, p1_d):
    rel_pos = rel_pos =  p1.unsqueeze(0) - p0.unsqueeze(1)
    z_distance = (rel_pos * p1_d).sum(-1, keepdim=True)
    z_displacement = z_distance * p1_d

    r_displacement = rel_pos - z_displacement
    r_distance = torch.norm(r_displacement, dim=-1, keepdim=True)
    return z_distance, r_distance

