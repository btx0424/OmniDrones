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


import abc
import os.path as osp
from contextlib import contextmanager
from typing import Dict, Sequence, Type

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.torch as torch_utils

import omni.timeline
import torch
from omni_drones.views import ArticulationView, RigidPrimView
from omni.isaac.core.simulation_context import SimulationContext
from torchrl.data import TensorSpec

import omni_drones.utils.kit as kit_utils

from omni_drones.robots.config import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
    RobotCfg,
)

ASSET_PATH = osp.join(osp.dirname(__file__), "assets")
TEMPLATE_PRIM_PATH = "/World/envs/env_0"


class RobotBase(abc.ABC):

    usd_path: str
    cfg_cls = RobotCfg

    _robots = {}
    _envs_positions: torch.Tensor

    REGISTRY: Dict[str, Type["RobotBase"]] = {}

    def __init__(self, name: str, cfg: RobotCfg = None, is_articulation=True) -> None:
        if name is None:
            name = self.__class__.__name__
        if name in RobotBase._robots:
            raise RuntimeError
        RobotBase._robots[name] = self
        if cfg is None:
            cfg = self.cfg_cls()
        
        self.name = name
        self.is_articulation = is_articulation
        self.rigid_props: RigidBodyPropertiesCfg = cfg.rigid_props
        self.articulation_props: ArticulationRootPropertiesCfg = cfg.articulation_props

        self.n = 0

        if SimulationContext._instance is None:
            raise RuntimeError("The SimulationContext is not created.")

        self.cfg = cfg
        self.device = SimulationContext.instance()._device
        self.dt = SimulationContext.instance().get_physics_dt()
        self.state_spec: TensorSpec
        self.action_spec: TensorSpec
        self.initialized = False

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if cls.__name__ in RobotBase.REGISTRY:
            raise ValueError
        super().__init_subclass__(**kwargs)
        RobotBase.REGISTRY[cls.__name__] = cls
        RobotBase.REGISTRY[cls.__name__.lower()] = cls

    def spawn(
        self, 
        translations=[(0.0, 0.0, 0.5)], 
        orientations=None,
        prim_paths: Sequence[str] = None
    ):
        if SimulationContext.instance()._physics_sim_view is not None:
            raise RuntimeError(
                "Cannot spawn robots after simulation_context.reset() is called."
            )
        
        translations = torch.atleast_2d(
            torch.as_tensor(translations, device=self.device)
        )
        n = translations.shape[0]

        if orientations is None:
            orientations = [None for _ in range(n)]

        if prim_paths is None:
            prim_paths = [f"{TEMPLATE_PRIM_PATH}/{self.name}_{i}" for i in range(n)]

        if not len(translations) == len(prim_paths):
            raise ValueError

        prims = []
        for prim_path, translation, orientation in zip(prim_paths, translations, orientations):
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate prim at {prim_path}.")
            prim = self._create_prim(prim_path, translation, orientation)
            # apply rigid body properties
            kit_utils.set_nested_rigid_body_properties(
                prim_path,
                linear_damping=self.rigid_props.linear_damping,
                angular_damping=self.rigid_props.angular_damping,
                max_linear_velocity=self.rigid_props.max_linear_velocity,
                max_angular_velocity=self.rigid_props.max_angular_velocity,
                max_depenetration_velocity=self.rigid_props.max_depenetration_velocity,
                enable_gyroscopic_forces=True,
                disable_gravity=self.rigid_props.disable_gravity,
                retain_accelerations=self.rigid_props.retain_accelerations,
            )
            # articulation root settings
            if self.is_articulation:
                kit_utils.set_articulation_properties(
                    prim_path,
                    enable_self_collisions=self.articulation_props.enable_self_collisions,
                    solver_position_iteration_count=self.articulation_props.solver_position_iteration_count,
                    solver_velocity_iteration_count=self.articulation_props.solver_velocity_iteration_count,
                )
            prims.append(prim)

        self.n += n
        return prims

    def _create_prim(self, prim_path, translation, orientation):
        prim = prim_utils.create_prim(
            prim_path,
            usd_path=self.usd_path,
            translation=translation,
            orientation=orientation,
        )
        return prim

    def initialize(
        self,
        prim_paths_expr: str = None,
    ):
        if SimulationContext.instance()._physics_sim_view is None:
            raise RuntimeError(
                f"Cannot initialize {self.__class__.__name__} before the simulation context resets."
                "Call simulation_context.reset() first."
            )
        if prim_paths_expr is None:
            prim_paths_expr = f"/World/envs/.*/{self.name}_*"
        self.prim_paths_expr = prim_paths_expr

        # create handles
        # -- robot articulation
        if self.is_articulation:
            self._view = ArticulationView(
                self.prim_paths_expr,
                reset_xform_properties=False,
                shape=(-1, self.n)
            )
            self.articulation = self
        else:
            self._view = RigidPrimView(
                self.prim_paths_expr, 
                reset_xform_properties=False,
                shape=(-1, self.n),
                # track_contact_forces=True
            )
            self.articulation = None
            self.articulation_indices = None

        self._view.initialize()
        # set the default state
        self._view.post_reset()
        self.shape = torch.arange(self._view.count).reshape(-1, self.n).shape
        
        self.prim_paths = self._view.prim_paths
        self.initialized = True

    @abc.abstractmethod
    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor, train: bool=True):
        raise NotImplementedError

    def get_world_poses(self, clone: bool=False):
        return self._view.get_world_poses(clone=clone)

    def set_world_poses(self, positions: torch.Tensor=None, orientations: torch.Tensor=None, env_indices: torch.Tensor = None):
        return self._view.set_world_poses(positions, orientations, env_indices=env_indices)

    def get_velocities(self, clone: bool=False):
        return self._view.get_velocities(clone=clone)

    def set_velocities(self, velocities: torch.Tensor, env_indices: torch.Tensor = None):
        return self._view.set_velocities(velocities, env_indices=env_indices)

    def get_joint_positions(self, clone: bool=False):
        if not self.is_articulation:
            raise NotImplementedError
        return self._view.get_joint_positions(clone=clone)

    def set_joint_positions(self, pos: torch.Tensor, env_indices: torch.Tensor = None):
        if not self.is_articulation:
            raise NotImplementedError
        return self._view.set_joint_positions(pos, env_indices=env_indices)

    def set_joint_position_targets(self, pos: torch.Tensor, env_indices: torch.Tensor = None):
        if not self.is_articulation:
            raise NotImplementedError
        self._view.set_joint_position_targets(pos, env_indices=env_indices)

    def get_joint_velocities(self, clone: bool=False):
        return self._view.get_joint_velocities(clone=clone)

    def set_joint_velocities(self, vel: torch.Tensor, env_indices: torch.Tensor = None):
        return self._view.set_joint_velocities(vel, env_indices=env_indices)

    def get_force_sensor_forces(self, clone: bool=False):
        if self.is_articulation:
            forces = self._view.get_force_sensor_forces(clone=clone)
        else:
            forces = self.articulation._view.get_force_sensor_forces(clone=clone)
            forces = forces[..., self.articulation_indices, :]
            forces = forces.reshape(*self.shape, 1, 6)
        return forces
    
    def get_state(self):
        raise NotImplementedError

