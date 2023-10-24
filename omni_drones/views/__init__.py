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
from typing import Optional, Tuple, List
from contextlib import contextmanager

from typing import List, Optional, Tuple, Union
import numpy as np
import carb
from omni.isaac.core.utils.prims import get_prim_parent, get_prim_at_path, set_prim_property, get_prim_property
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
from omni.isaac.core.utils.types import JointsState, ArticulationActions
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
        """Create a physics simulation view if not passed and creates an articulation view using physX tensor api.

        Args:
            physics_sim_view (omni.physics.tensors.SimulationView, optional): current physics simulation view. Defaults to None.
        """
        if physics_sim_view is None:
            physics_sim_view = omni.physics.tensors.create_simulation_view(self._backend)
            physics_sim_view.set_subspace_roots("/")
        carb.log_info("initializing view for {}".format(self._name))
        # TODO: add a callback to set physics view to None once stop is called
        self._physics_view = physics_sim_view.create_articulation_view(
            self._regex_prim_paths.replace(".*", "*"), self._enable_dof_force_sensors
        )
        assert self._physics_view.is_homogeneous
        self._physics_sim_view = physics_sim_view
        if not self._is_initialized:
            self._metadata = self._physics_view.shared_metatype
            self._num_dof = self._physics_view.max_dofs
            self._num_bodies = self._physics_view.max_links
            self._num_shapes = self._physics_view.max_shapes
            self._num_fixed_tendons = self._physics_view.max_fixed_tendons
            self._body_names = self._metadata.link_names
            self._body_indices = dict(zip(self._body_names, range(len(self._body_names))))
            self._dof_names = self._metadata.dof_names
            self._dof_indices = self._metadata.dof_indices
            self._dof_types = self._metadata.dof_types
            self._dof_paths = self._physics_view.dof_paths
            self._prim_paths = self._physics_view.prim_paths
            carb.log_info("Articulation Prim View Device: {}".format(self._device))
            self._is_initialized = True
            self._default_kps, self._default_kds = self.get_gains(clone=True)
            default_actions = self.get_applied_actions(clone=True)
            # TODO: implement effort part
            if self.num_dof > 0:
                if self._default_joints_state is None:
                    self._default_joints_state = JointsState(positions=None, velocities=None, efforts=None)
                if self._default_joints_state.positions is None:
                    self._default_joints_state.positions = default_actions.joint_positions
                if self._default_joints_state.velocities is None:
                    self._default_joints_state.velocities = default_actions.joint_velocities
                if self._default_joints_state.efforts is None:
                    self._default_joints_state.efforts = self._backend_utils.create_zeros_tensor(
                        shape=[self.count, self.num_dof], dtype="float32", device=self._device
                    )
        return
    
    def get_gains(
        self,
        indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        clone: bool = True,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Gets stiffness and damping of articulations in the view.

        Args:
            indices (Optional[Union[np.ndarray, List, torch.Tensor]], optional): indicies to specify which prims 
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor]], optional): joint indicies to specify which joints 
                                                                                 to query. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).
            clone (bool, optional): True to return clones of the internal buffers. Otherwise False. Defaults to True.

        Returns:
            Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]: stiffness and damping of
                                                             articulations in the view respectively. shapes are (M, K).
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return None
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, device="cpu")
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, device="cpu")
            if joint_indices.numel() == 0:
                return None, None
            kps = self._physics_view.get_dof_stiffnesses()
            kds = self._physics_view.get_dof_dampings()
            result_kps = self._backend_utils.move_data(
                kps[self._backend_utils.expand_dims(indices, 1), joint_indices], device=self._device
            )
            result_kds = self._backend_utils.move_data(
                kds[self._backend_utils.expand_dims(indices, 1), joint_indices], device=self._device
            )
            if clone:
                result_kps = self._backend_utils.clone_tensor(result_kps, device=self._device)
                result_kds = self._backend_utils.clone_tensor(result_kds, device=self._device)
            return result_kps, result_kds
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            dof_types = self.get_dof_types()
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            if joint_indices.numel() == 0:
                return None, None
            kps = self._backend_utils.create_zeros_tensor(
                shape=[indices.shape[0], joint_indices.shape[0]], dtype="float32", device=self._device
            )
            kds = self._backend_utils.create_zeros_tensor(
                shape=[indices.shape[0], joint_indices.shape[0]], dtype="float32", device=self._device
            )
            articulation_write_idx = 0
            for i in indices:
                dof_write_idx = 0
                for dof_index in joint_indices:
                    drive_type = (
                        "angular" if dof_types[dof_index] == omni.physics.tensors.DofType.Rotation else "linear"
                    )
                    prim = get_prim_at_path(self._dof_paths[i][dof_index])
                    if prim.HasAPI(UsdPhysics.DriveAPI):
                        drive = UsdPhysics.DriveAPI(prim, drive_type)
                    else:
                        drive = UsdPhysics.DriveAPI.Apply(prim, drive_type)
                    if drive.GetStiffnessAttr().Get() == 0.0 or drive_type == "linear":
                        kps[articulation_write_idx][dof_write_idx] = drive.GetStiffnessAttr().Get()
                    else:
                        kps[articulation_write_idx][dof_write_idx] = self._backend_utils.convert(
                            1.0 / omni.isaac.core.utils.numpy.deg2rad(float(1.0 / drive.GetStiffnessAttr().Get())),
                            device=self._device,
                        )
                    if drive.GetDampingAttr().Get() == 0.0 or drive_type == "linear":
                        kds[articulation_write_idx][dof_write_idx] = drive.GetDampingAttr().Get()
                    else:
                        kds[articulation_write_idx][dof_write_idx] = self._backend_utils.convert(
                            1.0 / omni.isaac.core.utils.numpy.deg2rad(float(1.0 / drive.GetDampingAttr().Get())),
                            device=self._device,
                        )
                    dof_write_idx += 1
                articulation_write_idx += 1
            return kps, kds
    
    def get_applied_actions(self, clone: bool = True) -> ArticulationActions:
        """Gets current applied actions in an ArticulationActions object.

        Args:
            clone (bool, optional): True to return clones of the internal buffers. Otherwise False. Defaults to True.

        Returns:
            ArticulationActions: current applied actions (i.e: current position targets and velocity targets)
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return None
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            if self.num_dof == 0:
                return None
            self._physics_sim_view.enable_warnings(False)
            joint_positions = self._physics_view.get_dof_position_targets()
            if clone:
                joint_positions = self._backend_utils.clone_tensor(joint_positions, device=self._device)
            joint_velocities = self._physics_view.get_dof_velocity_targets()
            if clone:
                joint_velocities = self._backend_utils.clone_tensor(joint_velocities, device=self._device)
            self._physics_sim_view.enable_warnings(True)
            # TODO: implement the effort part
            return ArticulationActions(
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                joint_efforts=None,
                joint_indices=None,
            )
        else:
            carb.log_warn("Physics Simulation View is not created yet in order to use get_applied_actions")
            return None

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
            velocities.flatten(end_dim=-2), 
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

    def get_dof_limits(self) -> torch.Tensor:
        return (
            super().get_dof_limits()
            .unflatten(0, self.shape)
            .to(self._device)
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

    def get_force_sensor_forces(self, env_indices: Optional[torch.Tensor] = None, clone: bool = False) -> torch.Tensor:
        with disable_warnings(self._physics_sim_view):
            forces = torch.unflatten(self._physics_view.get_force_sensor_forces(), 0, self.shape)
        if clone:
            forces = forces.clone()
        if env_indices is not None:
            forces = forces[env_indices]
        return forces

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
    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None):
        super().initialize(physics_sim_view)
        self.shape = torch.arange(self.count).reshape(self.shape).shape
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

    def get_net_contact_forces(
        self,
        env_indices: Optional[torch.Tensor] = None,
        clone: bool = False,
        dt: float = 1,
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return (
            super().get_net_contact_forces(indices, clone, dt).unflatten(0, self.shape)
        )

    def get_contact_force_matrix(
        self, 
        env_indices: Optional[torch.Tensor] = None, 
        clone: bool = True, 
        dt: float = 1
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().get_contact_force_matrix(indices, clone, dt).unflatten(0, self.shape)

    def get_masses(
        self, 
        env_indices: Optional[torch.Tensor] = None, 
        clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            current_values = self._backend_utils.move_data(self._physics_view.get_masses(), self._device)
            masses = current_values[indices]
            if clone:
                masses = self._backend_utils.clone_tensor(masses, device=self._device)
        else:
            masses = self._backend_utils.create_zeros_tensor([indices.shape[0]], dtype="float32", device=self._device)
            write_idx = 0
            for i in indices:
                if self._mass_apis[i.tolist()] is None:
                    if self._prims[i.tolist()].HasAPI(UsdPhysics.MassAPI):
                        self._mass_apis[i.tolist()] = UsdPhysics.MassAPI(self._prims[i.tolist()])
                    else:
                        self._mass_apis[i.tolist()] = UsdPhysics.MassAPI.Apply(self._prims[i.tolist()])
                masses[write_idx] = self._backend_utils.create_tensor_from_list(
                    self._mass_apis[i.tolist()].GetMassAttr().Get(), dtype="float32", device=self._device
                )
                write_idx += 1
        return masses.reshape(-1, *self.shape[1:], 1)
    
    def set_masses(
        self, 
        masses: torch.Tensor, 
        env_indices: Optional[torch.Tensor] = None
    ) -> None:
        indices = self._resolve_env_indices(env_indices).cpu()
        masses = masses.reshape(-1, 1)
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            data = self._backend_utils.clone_tensor(self._physics_view.get_masses(), device="cpu")
            data[indices] = self._backend_utils.move_data(masses, device="cpu")
            self._physics_view.set_masses(data, indices)
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            read_idx = 0
            for i in indices:
                if self._mass_apis[i.tolist()] is None:
                    if self._prims[i.tolist()].HasAPI(UsdPhysics.MassAPI):
                        self._mass_apis[i.tolist()] = UsdPhysics.MassAPI(self._prims[i.tolist()])
                    else:
                        self._mass_apis[i.tolist()] = UsdPhysics.MassAPI.Apply(self._prims[i.tolist()])
                self._mass_apis[i.tolist()].GetMassAttr().Set(masses[read_idx].tolist())
                read_idx += 1
            return

    def get_coms(
        self, 
        env_indices: Optional[torch.Tensor] = None, 
        clone: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self._resolve_env_indices(env_indices)
        positions, orientations = super().get_coms(indices, clone)
        return positions.unflatten(0, self.shape), orientations.unflatten(0, self.shape)
    
    def set_coms(
        self, 
        positions: torch.Tensor = None, 
        # orientations: torch.Tensor = None, 
        env_indices: torch.Tensor = None
    ) -> None:
        # TODO@btx0424 fix orientations
        indices = self._resolve_env_indices(env_indices)
        return super().set_coms(positions.reshape(-1, 1, 3), None, indices)
    
    def get_inertias(
        self, 
        env_indices: Optional[torch.Tensor]=None, 
        clone: bool=True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().get_inertias(indices, clone).unflatten(0, self.shape)
    
    def set_inertias(
        self, 
        values: torch.Tensor, 
        env_indices: Optional[torch.Tensor]=None
    ):
        indices = self._resolve_env_indices(env_indices)
        return super().set_inertias(values.reshape(-1, 9), indices)

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
