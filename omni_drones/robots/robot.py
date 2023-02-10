import torch
import os.path as osp

from torchrl.data import TensorSpec

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
    _robots = {}

    state_spec: TensorSpec
    action_spec: TensorSpec

    def __init__(self, name: str, cfg: RobotCfg=None) -> None:
        if name in RobotBase._robots:
            raise RuntimeError
        RobotBase._robots[name] = self
        if cfg is None:
            cfg = RobotCfg()
        self.name = name
        self.rigid_props: RigidBodyPropertiesCfg = cfg.rigid_props
        self.articulation_props: ArticulationRootPropertiesCfg = cfg.articulation_props
        self.count = 0
    
    def spawn(
        self, n: int=1, 
    ):
        if SimulationContext._instance._physics_sim_view is not None:
            raise RuntimeError(
                "Cannot spawn robots after simulation_context.reset() is called."
            )
        
        for i in range(self.count, self.count + n):
            prim_path = f"{TEMPLATE_PRIM_PATH}/{self.name}_{i}"
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(
                    f"Duplicate prim at {prim_path}."
                )
            prim = prim_utils.create_prim(
                prim_path,
                usd_path=self.usd_path,
            )
            prim.GetAllChildren
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

        self.count += n

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

    def apply_action(self, actions: torch.Tensor):
        raise NotImplementedError