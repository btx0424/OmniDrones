import yaml
import torch
import logging
from functorch import vmap
from tensordict.nn import make_functional
from typing import Type
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.prims import RigidPrimView

from omni_drones.robots import RobotBase
from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.controllers import LeePositionController

class MultirotorBase(RobotBase):

    param_path: str
    default_controller: Type = LeePositionController

    def __init__(self, name: str=None, cfg=None) -> None:
        super().__init__(name, cfg)

        with open(self.param_path, "r") as f:
            logging.info(f"Reading {name}'s params from {self.param_path}.")
            self.params = yaml.safe_load(f)
        self.mass = self.params["mass"]
        self.num_rotors = self.params["rotor_configuration"]["num_rotors"]

        self.action_spec = BoundedTensorSpec(-1, 1, self.num_rotors, device=self.device)
        self.state_spec = UnboundedContinuousTensorSpec(19 + self.num_rotors, device=self.device)
        self.prim_paths_expr = f"/World/envs/.*/{self.name}_*"
        

    def initialize(self):
        super().initialize()
        self.base_link = RigidPrimView(
            prim_paths_expr=f"{self.prim_paths_expr}/base_link", name="base_link")
        self.base_link.initialize()
        self.rotors_view = RigidPrimView(
            prim_paths_expr=f"{self.prim_paths_expr}/rotor_[0-{self.num_rotors-1}]", name="rotors")
        self.rotors_view.initialize()
        translate, _ = self.rotors_view.get_local_poses()
        arm_lengths = torch.norm(translate, dim=-1)
        print(arm_lengths)
        self.rotors = RotorGroup(self.params["rotor_configuration"], dt=self.dt).to(self.device)
        self.rotor_params_and_states = make_functional(self.rotors).expand(self.shape).clone()
        
        self.max_forces = self.rotor_params_and_states["max_forces"]
        self.throttle = self.rotor_params_and_states["throttle"]
        self.KF = self.rotor_params_and_states["KF"]
        self.KM = self.rotor_params_and_states["KM"]
        self.directions = self.rotor_params_and_states["directions"]
        
        self.forces = torch.zeros(*self.shape, self.num_rotors, 3, device=self.device)
        self.torques = torch.zeros(*self.shape, 3, device=self.device)

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds = actions.expand(*self.shape, self.num_rotors)
        thrusts, moments = vmap(vmap(self.rotors))(rotor_cmds, self.rotor_params_and_states)
        self.forces[..., 2] = thrusts
        self.torques[..., 2] = moments.sum(-1)
        self.rotors_view.apply_forces(self.forces.reshape(-1, 3), is_global=False)
        self.base_link.apply_forces_and_torques_at_pos(
            None, self.torques.reshape(-1, 3), is_global=False)
        return self.throttle.sum(-1)
    
    def get_state(self):
        pos, rot = self.get_env_poses(True)
        vel = self.get_velocities(True)
        thr = self.throttle * 2 - 1
        heading = vmap(torch_utils.quat_axis)(rot, axis=0)
        up = vmap(torch_utils.quat_axis)(rot, axis=2)
        return torch.cat([pos, rot, vel, heading, up, thr], dim=-1)
    
    def _reset_idx(self, env_ids: torch.Tensor):
        self.forces[env_ids] = 0.
        self.torques[env_ids] = 0.
        self.throttle[env_ids] = 0.
        
