import torch
import os.path as osp
from contextlib import contextmanager
from torchrl.data import TensorSpec

import omni.timeline
import omni.isaac.core.utils.prims as prim_utils
import omni_drones.utils.kit as kit_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.articulations import ArticulationView

from omni_drones.robots.config import (
    RobotCfg,
    RigidBodyPropertiesCfg,
    ArticulationRootPropertiesCfg,
)

ASSET_PATH = osp.join(osp.dirname(__file__), "assets")
TEMPLATE_PRIM_PATH = "/World/envs/env_0"

class RobotBase:

    usd_path: str
    state_spec: TensorSpec
    action_spec: TensorSpec

    _robots = {}
    _envs_positions: torch.Tensor

    def __init__(self, name: str, cfg: RobotCfg=None) -> None:
        if name in RobotBase._robots:
            raise RuntimeError
        RobotBase._robots[name] = self
        if cfg is None:
            cfg = RobotCfg()
        self.name = name
        self.rigid_props: RigidBodyPropertiesCfg = cfg.rigid_props
        self.articulation_props: ArticulationRootPropertiesCfg = cfg.articulation_props
        
        self._count = 0
    
    @property
    def device(self):
        device = SimulationContext._instance._device
        if device is None:
            raise RuntimeError("The SimulationContext is not created.")
        return device

    def spawn(
        self, n: int=1, translation=(0., 0., 0.5)
    ):
        if SimulationContext._instance._physics_sim_view is not None:
            raise RuntimeError(
                "Cannot spawn robots after simulation_context.reset() is called."
            )
        translation = torch.atleast_2d(torch.as_tensor(translation, device=self.device))
        if n != len(translation):
            raise ValueError
        for i in range(self._count, self._count + n):
            prim_path = f"{TEMPLATE_PRIM_PATH}/{self.name}_{i}"
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(
                    f"Duplicate prim at {prim_path}."
                )
            prim = prim_utils.create_prim(
                prim_path,
                usd_path=self.usd_path,
                translation=translation[i],
            )
            # apply rigid body properties
            kit_utils.set_nested_rigid_body_properties(
                prim_path,
                linear_damping=self.rigid_props.linear_damping,
                angular_damping=self.rigid_props.angular_damping,
                max_linear_velocity=self.rigid_props.max_linear_velocity,
                max_angular_velocity=self.rigid_props.max_angular_velocity,
                max_depenetration_velocity=self.rigid_props.max_depenetration_velocity,
                disable_gravity=self.rigid_props.disable_gravity,
                retain_accelerations=self.rigid_props.retain_accelerations,
            )
            # articulation root settings
            kit_utils.set_articulation_properties(
                prim_path,
                enable_self_collisions=self.articulation_props.enable_self_collisions,
                solver_position_iteration_count=self.articulation_props.solver_position_iteration_count,
                solver_velocity_iteration_count=self.articulation_props.solver_velocity_iteration_count,
            )

        self._count += n

    def initialize(self):
        if SimulationContext._instance._physics_sim_view is None:
            raise RuntimeError(
                "Cannot create ArticulationView before the simulation context resets."
                "Call simulation_context.reset() first."
            )
        prim_paths_expr = f"/World/envs/.*/{self.name}_*"
        # create handles
        # -- robot articulation
        self.articulations = ArticulationView(
            prim_paths_expr, reset_xform_properties=False
        )
        self.articulations.initialize()
        # set the default state
        self.articulations.post_reset()
        self.shape = (
            torch.arange(self.articulations.count)
            .reshape(-1, self._count).shape
        )
        self._physics_view = self.articulations._physics_view
        self._physics_sim_view = self.articulations._physics_sim_view
        
        pos, rot = self.get_world_poses()
        self.set_env_poses(pos, rot)

        print(self.articulations._dof_names)
        print(self.articulations._dof_types)
        print(self.articulations._dofs_infos)

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_world_poses(self):
        with self._disable_warnings():
            poses = torch.unflatten(self._physics_view.get_root_transforms(), 0, self.shape)
        return poses[..., :3], poses[..., 3:7]

    def get_env_poses(self):
        with self._disable_warnings():
            poses = torch.unflatten(self._physics_view.get_root_transforms(), 0, self.shape)
            poses[..., :3] -= self._envs_positions
        return poses[..., :3], poses[..., 3:7]
        
    def set_env_poses(self, 
        positions: torch.Tensor, 
        orientations: torch.Tensor,
        indices: torch.Tensor=None,
    ):
        with self._disable_warnings():
            positions += self._envs_positions
            pose = torch.cat([positions, orientations], dim=-1)
            all_indices = torch.arange(self.articulations.count, device=self.device).reshape(self.shape)
            if indices is None:
                indices = all_indices
            else:
                indices = all_indices[indices]
            self._physics_view.set_root_transforms(pose.flatten(0, -2), indices.flatten())

    def get_velocities(self):
        with self._disable_warnings():
            velocities = torch.unflatten(self._physics_view.get_root_velocities(), 0, self.shape)
        return velocities

    @contextmanager
    def _disable_warnings(self):
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            try:
                self._physics_sim_view.enable_warnings(False)
                yield
            finally:
                self._physics_sim_view.enable_warnings(True)
        else:
            raise RuntimeError

    