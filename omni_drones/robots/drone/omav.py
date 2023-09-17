# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from functorch import vmap
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.robots.drone import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH

class Omav(MultirotorBase):

    usd_path: str = ASSET_PATH + "/usd/omav.usd"
    param_path: str = ASSET_PATH + "/usd/omav.yaml"

    def __init__(self, name: str = "Omav", cfg=None) -> None:
        super().__init__(name, cfg)
        self.action_spec = BoundedTensorSpec(-1, 1, 12 + 6, device=self.device)
        self.tilt_dof_indices = torch.arange(0, 6, device=self.device)
        self.rotor_dof_indices = torch.arange(6, 18, device=self.device)
        self.max_tilt_velocity = 10

        self.action_spec = BoundedTensorSpec(-1, 1, self.num_rotors + 6, device=self.device)
        self.state_spec = UnboundedContinuousTensorSpec(
            19 + self.num_rotors + 12, device=self.device
        )

    def initialize(self, prim_paths_expr: str = None):
        if not self.is_articulation:
            raise NotImplementedError
        super().initialize(prim_paths_expr)
        self.init_joint_positions = self._view.get_joint_positions()
        self.init_joint_velocities = self._view.get_joint_velocities()
    
    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds, tilt_cmds = actions.expand(*self.shape, 18).split([12, 6], dim=-1)
        super().apply_action(rotor_cmds)

        velocity_targets = tilt_cmds.clamp(-1, 1) * self.max_tilt_velocity
        self._view.set_joint_velocity_targets(
            velocity_targets, joint_indices=self.tilt_dof_indices
        )

        return self.throttle.sum(-1)

    def _reset_idx(self, env_ids: torch.Tensor, train: bool=True):
        env_ids = super()._reset_idx(env_ids, train)
        self._view.set_joint_positions(
            self.init_joint_positions[env_ids],
            env_indices=env_ids,
        )
        self._view.set_joint_velocities(
            self.init_joint_velocities[env_ids],
            env_indices=env_ids,
        )
        return env_ids

    def get_state(self, env=True):
        state = super().get_state(env)
        joint_positions = self.get_joint_positions()[..., self.tilt_dof_indices]
        joint_velocities = self.get_joint_velocities()[..., self.tilt_dof_indices]
        state = torch.cat([state, joint_positions, joint_velocities], dim=-1)
        assert not torch.isnan(state).any()
        return state

