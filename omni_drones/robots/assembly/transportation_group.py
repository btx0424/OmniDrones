from typing import Sequence, Union

import omni.isaac.core.objects as objects
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import torch

from omni.isaac.core.prims import RigidPrimView
from omni.kit.commands import execute
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils

from omni_drones.robots import ASSET_PATH, RobotBase
from omni_drones.robots.drone import MultirotorBase


class TransportationGroup(RobotBase):

    def __init__(
        self,
        name: str = "Group",
        drone: Union[str, MultirotorBase] = "Firefly",
        cfg=None,
        is_articulation=True,
    ) -> None:
        super().__init__(name, cfg, is_articulation)
        if isinstance(drone, str):
            drone = MultirotorBase.REGISTRY[drone]()
        drone.is_articulation = False
        self.drone = drone
        self.translations = []

    def spawn(
        self, 
        translations=..., 
        prim_paths: Sequence[str] = None,
        enable_collision: bool = False,
    ):

        translations = torch.atleast_2d(
            torch.as_tensor(translations, device=self.device)
        )
        self.translations.extend(translations.tolist())
        n = translations.shape[0]

        if prim_paths is None:
            prim_paths = [f"/World/envs/env_0/{self.name}_{i}" for i in range(n)]

        prims = []
        for prim_path, translation in zip(prim_paths, translations):
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate prim at {prim_path}.")
            xform = prim_utils.create_prim(
                prim_path,
                translation=translation,
            )

            payload = prim_utils.create_prim(
                prim_path=f"{prim_path}/payload",
                prim_type="Cube",
                translation=(0.0, 0.0, -1.1),
                scale=(0.75, 0.5, 0.2),
            )

            script_utils.setRigidBody(payload, "convexHull", False)
            UsdPhysics.MassAPI.Apply(payload)
            payload.GetAttribute("physics:mass").Set(2.0)
            payload.GetAttribute("physics:collisionEnabled").Set(enable_collision)
            
            kit_utils.set_rigid_body_properties(
                payload.GetPath(),
                angular_damping=0.1,
                linear_damping=0.1
            )

            drone_translations = torch.tensor(
                [
                    [0.75, 0.5, 0],
                    [0.75, -0.5, 0],
                    [-0.75, -0.5, 0],
                    [-0.75, 0.5, 0],
                ]
            )

            for i in range(4):
                drone_prim = self.drone.spawn(
                    translations=drone_translations[i],
                    prim_paths=[f"{prim_path}/{self.drone.name.lower()}_{i}"],
                )[0]
                execute(
                    "UnapplyAPISchema",
                    api=UsdPhysics.ArticulationRootAPI,
                    prim=drone_prim,
                )
                execute(
                    "UnapplyAPISchema",
                    api=PhysxSchema.PhysxArticulationAPI,
                    prim=drone_prim,
                )

                scene_utils.create_bar(
                    prim_path=f"{prim_path}/{self.drone.name.lower()}_{i}/bar",
                    length=1,
                    translation=(0, 0, -0.5),
                    from_prim=payload,
                    to_prim=f"{prim_path}/{self.drone.name.lower()}_{i}/base_link",
                    mass=0.03,
                    enable_collision=enable_collision
                )

            UsdPhysics.ArticulationRootAPI.Apply(xform)
            PhysxSchema.PhysxArticulationAPI.Apply(xform)

            if self.is_articulation:
                kit_utils.set_articulation_properties(
                    xform.GetPath(),
                    enable_self_collisions=False,
                    solver_position_iteration_count=self.articulation_props.solver_position_iteration_count,
                    solver_velocity_iteration_count=self.articulation_props.solver_velocity_iteration_count,
                )
            prims.append(xform)

        self.n += n
        return prims

    def initialize(self, prim_paths_expr: str = None, track_contact_forces: bool = False):
        super().initialize(prim_paths_expr)
        self.drone.n = self.n * 4
        self.drone.initialize(f"{self.prim_paths_expr}/{self.drone.name.lower()}_*")
        self.payload_view = RigidPrimView(
            f"{self.prim_paths_expr}/payload",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces
        )
        self.payload_view.initialize()

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        self.drone.apply_action(actions)

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
