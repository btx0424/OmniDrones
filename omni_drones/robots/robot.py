import abc
import os.path as osp
from contextlib import contextmanager
from typing import Dict, Sequence, Type

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.torch as torch_utils

import omni.timeline
import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
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

    _robots = {}
    _envs_positions: torch.Tensor = None

    REGISTRY: Dict[str, Type["RobotBase"]] = {}

    def __init__(self, name: str, cfg: RobotCfg = None, is_articulation=True) -> None:
        if name is None:
            name = self.__class__.__name__
        if name in RobotBase._robots:
            raise RuntimeError
        RobotBase._robots[name] = self
        if cfg is None:
            cfg = RobotCfg()
        self.name = name
        self.is_articulation = is_articulation
        self.rigid_props: RigidBodyPropertiesCfg = cfg.rigid_props
        self.articulation_props: ArticulationRootPropertiesCfg = cfg.articulation_props

        self.n = 0

        if SimulationContext._instance is None:
            raise RuntimeError("The SimulationContext is not created.")

        self.device = SimulationContext.instance()._device
        self.dt = SimulationContext.instance().get_physics_dt()
        self.state_spec: TensorSpec
        self.action_spec: TensorSpec

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if cls.__name__ in RobotBase.REGISTRY:
            raise ValueError
        super().__init_subclass__(**kwargs)
        RobotBase.REGISTRY[cls.__name__] = cls
        RobotBase.REGISTRY[cls.__name__.lower()] = cls

    def spawn(self, translations=[(0.0, 0.0, 0.5)], prim_paths: Sequence[str] = None):
        if SimulationContext.instance()._physics_sim_view is not None:
            raise RuntimeError(
                "Cannot spawn robots after simulation_context.reset() is called."
            )
        translations = torch.atleast_2d(
            torch.as_tensor(translations, device=self.device)
        )
        n = translations.shape[0]

        if prim_paths is None:
            prim_paths = [f"{TEMPLATE_PRIM_PATH}/{self.name}_{i}" for i in range(n)]

        if not len(translations) == len(prim_paths):
            raise ValueError

        prims = []
        for prim_path, translation in zip(prim_paths, translations):
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate prim at {prim_path}.")
            prim = prim_utils.create_prim(
                prim_path,
                usd_path=self.usd_path,
                translation=translation,
            )
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
                    enable_self_collisions=False,
                    solver_position_iteration_count=self.articulation_props.solver_position_iteration_count,
                    solver_velocity_iteration_count=self.articulation_props.solver_velocity_iteration_count,
                )
            prims.append(prim)

        self.n += n
        return prims

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
            )
        else:
            self._view = RigidPrimView(
                self.prim_paths_expr, reset_xform_properties=False
            )

        self._view.initialize()
        # set the default state
        self._view.post_reset()
        self.shape = torch.arange(self._view.count).reshape(-1, self.n).shape
        self._physics_view = self._view._physics_view
        self._physics_sim_view = self._view._physics_sim_view

    @abc.abstractmethod
    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        raise NotImplementedError

    def get_world_poses(self, clone=True):
        with self._disable_warnings():
            if self.is_articulation:
                poses = torch.unflatten(
                    self._physics_view.get_root_transforms(), 0, self.shape
                )
            else:
                poses = torch.unflatten(
                    self._physics_view.get_transforms(), 0, self.shape
                )
            if clone:
                poses = poses.clone()
        return poses[..., :3], poses[..., [6, 3, 4, 5]]

    def get_env_poses(self, clone=True):
        with self._disable_warnings():
            if self.is_articulation:
                poses = torch.unflatten(
                    self._physics_view.get_root_transforms(), 0, self.shape
                )
            else:
                poses = torch.unflatten(
                    self._physics_view.get_transforms(), 0, self.shape
                )
            if clone:
                poses = poses.clone()
        if self._envs_positions is not None:
            poses[..., :3] -= self._envs_positions
        return poses[..., :3], poses[..., [6, 3, 4, 5]]

    def set_env_poses(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        indices: torch.Tensor = None,
    ):
        with self._disable_warnings():
            if self._envs_positions is not None:
                positions = (positions + self._envs_positions[indices]).reshape(-1, 3)
            else:
                positions = positions.reshape(-1, 3)
            orientations = orientations.reshape(-1, 4)[:, [1, 2, 3, 0]]
            if self.is_articulation:
                old_pose = self._physics_view.get_root_transforms().clone()
            else:
                old_pose = self._physics_view.get_transforms().clone()
            indices = self._resolve_indices(indices)
            if positions is None:
                positions = old_pose[indices, :3]
            if orientations is None:
                orientations = old_pose[indices, 3:]
            new_pose = torch.cat([positions, orientations], dim=-1)
            old_pose[indices] = new_pose
            if self.is_articulation:
                self._physics_view.set_root_transforms(old_pose, indices)
            else:
                self._physics_view.set_transforms(old_pose, indices)

    def get_velocities(self, clone=True):
        with self._disable_warnings():
            if self.is_articulation:
                velocities = torch.unflatten(
                    self._physics_view.get_root_velocities(), 0, self.shape
                )
            else:
                velocities = torch.unflatten(
                    self._physics_view.get_velocities(), 0, self.shape
                )
            if clone:
                velocities = velocities.clone()
        return velocities

    def set_velocities(self, velocities: torch.Tensor, indices: torch.Tensor = None):
        with self._disable_warnings():
            velocities = velocities.flatten(0, -2)
            indices = self._resolve_indices(indices)
            root_vel = self._physics_view.get_root_velocities()
            root_vel[indices] = velocities
            self._physics_view.set_root_velocities(root_vel, indices)

    def get_joint_positions(self, clone=True):
        with self._disable_warnings():
            joint_positions = torch.unflatten(
                self._physics_view.get_dof_positions(), 0, self.shape
            )
            if clone:
                joint_positions = joint_positions.clone()
        return joint_positions

    def set_joint_positions(self, pos: torch.Tensor, indices: torch.Tensor = None):
        with self._disable_warnings():
            pos = pos.flatten(0, -2)
            indices = self._resolve_indices(indices)
            joint_pos = self._physics_view.get_dof_positions()
            joint_pos[indices] = pos
            self._physics_view.set_dof_positions(pos, indices)

    def get_joint_velocities(self, clone=True):
        with self._disable_warnings():
            joint_velocities = torch.unflatten(
                self._physics_view.get_dof_velocities(), 0, self.shape
            )
            if clone:
                joint_velocities = joint_velocities.clone()
        return joint_velocities

    def set_joint_velocities(self, vel: torch.Tensor):
        with self._disable_warnings():
            vel = vel.flatten(0, -2)
            indices = self._resolve_indices(indices)
            joint_vel = self._physics_view.get_dof_velocities()
            joint_vel[indices] = vel
            self._physics_view.set_dof_velocities(vel)

    def get_state(self):
        raise NotImplementedError

    def _resolve_indices(self, indices: torch.Tensor = None):
        all_indices = torch.arange(self._view.count, device=self.device).reshape(
            self.shape
        )
        if indices is None:
            indices = all_indices
        else:
            indices = all_indices[indices]
        return indices.flatten()

    @contextmanager
    def _disable_warnings(self):
        if (
            not omni.timeline.get_timeline_interface().is_stopped()
            and self._physics_view is not None
        ):
            try:
                self._physics_sim_view.enable_warnings(False)
                yield
            finally:
                self._physics_sim_view.enable_warnings(True)
        else:
            raise RuntimeError
