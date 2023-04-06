import torch
from typing import Optional, Tuple, List
from contextlib import contextmanager

from omni.isaac.core.articulations import ArticulationView as _ArticulationView
from omni.isaac.core.prims import RigidPrimView as _RigidPrimView
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.simulation_context import SimulationContext
import omni
import functools


def require_sim_initialized(func):

    @functools.wraps(func)
    def _func(*args, **kwargs):
        if SimulationContext.instance()._physics_sim_view is None:
            raise RuntimeError("SimulationContext not initialzed.")
        return func(*args, **kwargs)
    
    return _func


class ArticulationView(_ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: str = "articulation_prim_view",
        positions: Optional[torch.Tensor] = None,
        translations: Optional[torch.Tensor] = None,
        orientations: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        visibilities: Optional[torch.Tensor] = None,
        reset_xform_properties: bool = True,
        enable_dof_force_sensors: bool = False,
        shape: Tuple[int, ...] = (-1,),
    ) -> None:
        self.shape = shape
        super().__init__(
            prim_paths_expr,
            name,
            positions,
            translations,
            orientations,
            scales,
            visibilities,
            reset_xform_properties,
            enable_dof_force_sensors,
        )
    
    @require_sim_initialized
    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None) -> None:
        super().initialize(physics_sim_view)
        return self

    def get_world_poses(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self._resolve_env_indices(env_indices)
        if self._physics_view is not None:
            with disable_warnings(self._physics_sim_view):
                poses = self._physics_view.get_root_transforms()[indices]
                poses = torch.unflatten(poses, 0, self.shape)
            if clone:
                poses = poses.clone()
            return poses[..., :3], poses[..., [6, 3, 4, 5]]
        else:
            pos, rot = super().get_world_poses(indices, clone)
            return pos.unflatten(0, self.shape), rot.unflatten(0, self.shape)

    def set_world_poses(
        self,
        positions: Optional[torch.Tensor] = None,
        orientations: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
    ) -> None:
        with disable_warnings(self._physics_sim_view):
            indices = self._resolve_env_indices(env_indices)
            poses = self._physics_view.get_root_transforms()
            if positions is not None:
                poses[indices, :3] = positions.reshape(-1, 3)
            if orientations is not None:
                poses[indices, 3:] = orientations.reshape(-1, 4)[:, [1, 2, 3, 0]]
            self._physics_view.set_root_transforms(poses, indices)

    def get_velocities(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().get_velocities(indices, clone).unflatten(0, self.shape)

    def set_velocities(
        self, velocities: torch.Tensor, env_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().set_velocities(velocities.reshape(-1, 6), indices)

    def get_joint_velocities(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return (
            super().get_joint_velocities(indices, clone=clone).unflatten(0, self.shape)
        )

    def set_joint_velocities(
        self,
        velocities: Optional[torch.Tensor],
        env_indices: Optional[torch.Tensor] = None,
        joint_indices: Optional[torch.Tensor] = None,
    ) -> None:
        indices = self._resolve_env_indices(env_indices)
        super().set_joint_velocities(
            velocities.reshape(-1, self.num_dof), 
            indices,
            joint_indices
        )

    def set_joint_velocity_targets(
        self, 
        velocities: Optional[torch.Tensor], 
        env_indices: Optional[torch.Tensor] = None, 
        joint_indices: Optional[torch.Tensor] = None
    ) -> None:
        indices = self._resolve_env_indices(env_indices)
        super().set_joint_velocity_targets(
            velocities.flatten(end_dim=-2), 
            indices, 
            joint_indices
        )

    def get_joint_positions(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return (
            super().get_joint_positions(indices, clone=clone).unflatten(0, self.shape)
        )

    def set_joint_positions(
        self,
        positions: Optional[torch.Tensor],
        env_indices: Optional[torch.Tensor] = None,
        joint_indices: Optional[torch.Tensor] = None,
    ) -> None:
        indices = self._resolve_env_indices(env_indices)
        super().set_joint_positions(
            positions.flatten(end_dim=-2), 
            indices,
            joint_indices
        )

    def set_joint_position_targets(
        self, 
        positions: Optional[torch.Tensor], 
        env_indices: Optional[torch.Tensor] = None, 
        joint_indices: Optional[torch.Tensor] = None,
    ) -> None:
        indices = self._resolve_env_indices(env_indices)
        super().set_joint_position_targets(
            positions.flatten(end_dim=-2), 
            indices,
            joint_indices
        )
    
    def set_joint_efforts(
        self, 
        efforts: Optional[torch.Tensor], 
        env_indices: Optional[torch.Tensor] = None, 
        joint_indices: Optional[torch.Tensor] = None
    ) -> None:
        indices = self._resolve_env_indices(env_indices)
        super().set_joint_efforts(
            efforts.flatten(end_dim=-2), 
            indices, 
            joint_indices
        )

    def get_body_masses(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().get_body_masses(indices, clone=clone).unflatten(0, self.shape)

    def set_body_masses(
        self,
        values: torch.Tensor,
        env_indices: Optional[torch.Tensor] = None,
    ) -> None:
        indices = self._resolve_env_indices(env_indices)
        return super().set_body_masses(values.reshape(-1, self.num_bodies), indices)

    def _resolve_env_indices(self, env_indices: torch.Tensor):
        if not hasattr(self, "_all_indices"):
            self._all_indices = torch.arange(self.count, device=self._device)
            self.shape = self._all_indices.reshape(self.shape).shape
        if env_indices is not None:
            indices = self._all_indices.reshape(self.shape)[env_indices].flatten()
        else:
            indices = self._all_indices
        return indices

    def squeeze_(self, dim: int = None):
        self.shape = self._all_indices.reshape(self.shape).squeeze(dim).shape
        return self


class RigidPrimView(_RigidPrimView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: str = "rigid_prim_view",
        positions: Optional[torch.Tensor] = None,
        translations: Optional[torch.Tensor] = None,
        orientations: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        visibilities: Optional[torch.Tensor] = None,
        reset_xform_properties: bool = True,
        masses: Optional[torch.Tensor] = None,
        densities: Optional[torch.Tensor] = None,
        linear_velocities: Optional[torch.Tensor] = None,
        angular_velocities: Optional[torch.Tensor] = None,
        track_contact_forces: bool = False,
        prepare_contact_sensors: bool = True,
        disable_stablization: bool = True,
        contact_filter_prim_paths_expr: Optional[List[str]] = (),
        shape: Tuple[int, ...] = (-1,),
    ) -> None:
        self.shape = shape
        super().__init__(
            prim_paths_expr,
            name,
            positions,
            translations,
            orientations,
            scales,
            visibilities,
            reset_xform_properties,
            masses,
            densities,
            linear_velocities,
            angular_velocities,
            track_contact_forces,
            prepare_contact_sensors,
            disable_stablization,
            contact_filter_prim_paths_expr,
        )

    @require_sim_initialized
    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None) -> None:
        super().initialize(physics_sim_view)
        return self

    def get_world_poses(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self._resolve_env_indices(env_indices)
        pos, rot = super().get_world_poses(indices, clone)
        return pos.unflatten(0, self.shape), rot.unflatten(0, self.shape)

    def set_world_poses(
        self,
        positions: Optional[torch.Tensor] = None,
        orientations: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
    ) -> None:
        with disable_warnings(self._physics_sim_view):
            indices = self._resolve_env_indices(env_indices)
            poses = self._physics_view.get_transforms()
            if positions is not None:
                poses[indices, :3] = positions.reshape(-1, 3)
            if orientations is not None:
                poses[indices, 3:] = orientations.reshape(-1, 4)[:, [1, 2, 3, 0]]
            self._physics_view.set_transforms(poses, indices)

    def get_velocities(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().get_velocities(indices, clone).unflatten(0, self.shape)

    def set_velocities(
        self, velocities: torch.Tensor, env_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().set_velocities(velocities.reshape(-1, 6), indices)

    # def get_net_contact_forces(
    #     self,
    #     env_indices: Optional[torch.Tensor] = None,
    #     clone: bool = False,
    #     dt: float = 1,
    # ) -> torch.Tensor:
    #     indices = self._resolve_env_indices(env_indices)
    #     return (
    #         super().get_net_contact_forces(indices.clone(), clone, dt).unflatten(0, self.shape)
    #     )

    def _resolve_env_indices(self, env_indices: torch.Tensor):
        if not hasattr(self, "_all_indices"):
            self._all_indices = torch.arange(self.count, device=self._device)
            self.shape = self._all_indices.reshape(self.shape).shape
        if env_indices is not None:
            indices = self._all_indices.reshape(self.shape)[env_indices].flatten()
        else:
            indices = self._all_indices
        return indices

    def squeeze_(self, dim: int = None):
        self.shape = self._all_indices.reshape(self.shape).squeeze(dim).shape
        return self


@contextmanager
def disable_warnings(physics_sim_view):
    try:
        physics_sim_view.enable_warnings(False)
        yield
    finally:
        physics_sim_view.enable_warnings(True)
