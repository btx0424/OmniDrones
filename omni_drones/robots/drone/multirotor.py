import logging
from typing import Type

import omni.isaac.core.utils.torch as torch_utils
import torch
import yaml
from functorch import vmap
from omni.isaac.core.prims import RigidPrimView
from tensordict.nn import make_functional
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec

from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.controllers import LeePositionController

from omni_drones.robots import RobotBase


class MultirotorBase(RobotBase):

    param_path: str
    DEFAULT_CONTROLLER: Type = LeePositionController

    def __init__(self, name: str = None, cfg=None) -> None:
        super().__init__(name, cfg)

        with open(self.param_path, "r") as f:
            logging.info(f"Reading {self.name}'s params from {self.param_path}.")
            self.params = yaml.safe_load(f)
        self.mass = self.params["mass"]
        self.num_rotors = self.params["rotor_configuration"]["num_rotors"]

        self.action_spec = BoundedTensorSpec(-1, 1, self.num_rotors, device=self.device)
        self.state_spec = UnboundedContinuousTensorSpec(
            19 + self.num_rotors, device=self.device
        )

    def initialize(self, prim_paths_expr: str = None):
        if self.is_articulation:
            super().initialize(prim_paths_expr=prim_paths_expr)
            self.base_link = RigidPrimView(
                prim_paths_expr=f"{self.prim_paths_expr}/base_link",
                name="base_link",
                track_contact_forces=True,
            )
            self.base_link.initialize()
        else:
            super().initialize(prim_paths_expr=f"{prim_paths_expr}/base_link")
            self.base_link = self._view
            self.prim_paths_expr = prim_paths_expr

        self.rotors_view = RigidPrimView(
            prim_paths_expr=f"{self.prim_paths_expr}/rotor_[0-{self.num_rotors-1}]",
            name="rotors",
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

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds = actions.expand(*self.shape, self.num_rotors)
        thrusts, moments = vmap(vmap(self.rotors))(
            rotor_cmds, self.rotor_params_and_states
        )
        self.forces[..., 2] = thrusts
        self.torques[..., 2] = moments.sum(-1)
        # self.articulations.set_joint_velocities(
        #     (self.throttle * self.directions * self.MAX_ROT_VEL).reshape(-1, self.num_rotors)
        # )
        self.rotors_view.apply_forces(self.forces.reshape(-1, 3), is_global=False)
        self.base_link.apply_forces_and_torques_at_pos(
            None, self.torques.reshape(-1, 3), is_global=False
        )
        return self.throttle.sum(-1)

    def get_state(self):
        pos, rot = self.get_env_poses(True)
        vel = self.get_velocities(True)
        thr = self.throttle * 2 - 1
        heading = vmap(torch_utils.quat_axis)(rot, axis=0)
        up = vmap(torch_utils.quat_axis)(rot, axis=2)
        return torch.cat([pos, rot, vel, heading, up, thr], dim=-1)

    def _reset_idx(self, env_ids: torch.Tensor):
        self.forces[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.throttle[env_ids] = 0.0
