from typing import Sequence, Union, Optional
from dataclasses import dataclass

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni_drones.utils.kit as kit_utils
import omni.physx.scripts.utils as script_utils
from omni.kit.commands import execute
import torch

from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
from scipy.spatial.transform.rotation import Rotation

from omni_drones.robots import RobotBase, RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView

def create_frame(
    prim_path: str,
    arm_angles: Sequence[float],
    arm_lengths: Sequence[float],
    to_prim_paths: Optional[Sequence[str]]=None,
    joint_damping: float = 0.002,
    color: Sequence[float] = (0.5, 0.5, 0.2),
    enable_collision: bool = False,
    exclude_from_articulation: bool = False,
):
    if isinstance(arm_angles, torch.Tensor):
        arm_angles = arm_angles.tolist()
    if isinstance(arm_lengths, torch.Tensor):
        arm_lengths = arm_lengths.tolist()

    stage = stage_utils.get_current_stage()
    prim_xform = prim_utils.create_prim(prim_path, translation=(0., 0., 0.))
    
    arms = []
    if to_prim_paths is None:
        to_prim_paths = [None for _ in range(len(arm_angles))]

    if not len(arm_angles) == len(arm_lengths) == len(to_prim_paths):
        raise ValueError
    
    for i, (arm_angle, arm_length, to_prim_path) in enumerate(
        zip(arm_angles, arm_lengths, to_prim_paths)
    ):
        link_path = f"{prim_path}/arm_{i}"
        capsuleGeom = UsdGeom.Capsule.Define(stage, link_path)

        capsuleGeom.CreateHeightAttr(arm_length)
        capsuleGeom.CreateRadiusAttr(arm_length * 0.02)
        capsuleGeom.CreateAxisAttr("X")

        r = Rotation.from_euler("z", arm_angle)
        location = Gf.Vec3f(*r.apply((arm_length / 2, 0, 0)))
        orient = Gf.Quatf(*r.as_quat()[[3, 0, 1, 2]])
        capsuleGeom.AddTranslateOp().Set(location)
        capsuleGeom.AddOrientOp().Set(orient)
        capsuleGeom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        capsuleGeom.CreateDisplayColorAttr().Set([color])

        UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
        prim: Usd.Prim = capsuleGeom.GetPrim()
        prim.GetAttribute("physics:collisionEnabled").Set(enable_collision)

        arms.append(prim)

        if to_prim_path is not None:
            to_prim = prim_utils.get_prim_at_path(to_prim_path)
            joint: Usd.Prim = script_utils.createJoint(stage, "D6", prim_xform, to_prim)
            rel_orient = joint.GetAttribute("physics:localRot0").Get()
            joint.GetAttribute("physics:localRot0").Set(orient)
            joint.GetAttribute("physics:localRot1").Set(rel_orient * orient)
            joint.GetAttribute("limit:rotX:physics:low").Set(-torch.inf)
            joint.GetAttribute("limit:rotX:physics:high").Set(torch.inf)
            joint.GetAttribute("limit:rotY:physics:low").Set(-torch.inf)
            joint.GetAttribute("limit:rotY:physics:high").Set(torch.inf)
            joint.GetAttribute("physics:excludeFromArticulation").Set(exclude_from_articulation)

            UsdPhysics.DriveAPI.Apply(joint, "rotX")
            UsdPhysics.DriveAPI.Apply(joint, "rotY")
            joint.GetAttribute("drive:rotX:physics:damping").Set(joint_damping)
            joint.GetAttribute("drive:rotY:physics:damping").Set(joint_damping)

    script_utils.setRigidBody(prim_xform, "convexHull", False)
    UsdPhysics.RigidBodyAPI.Apply(prim_xform)
    massAPI = UsdPhysics.MassAPI.Apply(prim_xform)
    massAPI.CreateMassAttr().Set(0.2)

    for arm in arms:
        arm.GetAttribute("physics:collisionEnabled").Set(enable_collision)
    return prim_xform


@dataclass
class PlatformCfg(RobotCfg):
    num_drones: int = 4
    rotate_drones: bool = True
    arm_length: float = 0.85
    # frame_mass: float = 0.2
    joint_damping: float = 0.001

class OveractuatedPlatform(RobotBase):

    def __init__(
        self,
        name: str="Platform",
        drone: Union[str, RobotBase]="Hummingbird",
        cfg: PlatformCfg=None,
        is_articulation: bool=True
    ):
        super().__init__(name, cfg, is_articulation)
        drone.is_articulation = False
        self.drone = drone

        self.joint_damping = cfg.joint_damping
        self.rotate_drones = cfg.rotate_drones
        self.arm_angles = torch.linspace(0, torch.pi*2, cfg.num_drones+1)[:-1]
        self.arm_lengths = torch.ones(cfg.num_drones) * cfg.arm_length

    def spawn(
        self, 
        translations=..., 
        prim_paths: Sequence[str] = None,
        enable_collision: bool = False,
    ):
        translations = torch.atleast_2d(
            torch.as_tensor(translations, device=self.device)
        )
        n = translations.shape[0]

        if prim_paths is None:
            prim_paths = [f"/World/envs/env_0/{self.name}_{i}" for i in range(n)]
        
        prims = []
        for i, (prim_path, translation) in enumerate(zip(prim_paths, translations)):
            
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate prim at {prim_path}.")

            xform = prim_utils.create_prim(
                prim_path,
                translation=translation,
            )

            drone_translations = torch.stack([
                torch.cos(self.arm_angles), 
                torch.sin(self.arm_angles), 
                torch.zeros_like(self.arm_angles)
            ], dim=-1) * self.arm_lengths.unsqueeze(1)

            drone_rotations = torch.tensor(
                Rotation.from_euler("z", -self.arm_angles)
                .as_quat()[:, [3, 0, 1, 2]]
            )

            drone_prims = self.drone.spawn(
                translations=drone_translations,
                orientations=drone_rotations if self.rotate_drones else None,
                prim_paths=[
                    f"/World/envs/env_0/{self.name}_{i}/{self.drone.name}_{j}" 
                    for j in range(drone_translations.shape[0])
                ],
            )
            for drone_prim in drone_prims:
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

            self._create_frame(
                f"/World/envs/env_0/{self.name}_{i}/frame",
                [
                    f"{drone_prim.GetPath().pathString}/base_link"
                    for drone_prim in drone_prims
                ],
                enable_collision=enable_collision,
            )
            UsdPhysics.ArticulationRootAPI.Apply(xform)
            PhysxSchema.PhysxArticulationAPI.Apply(xform)

            kit_utils.set_nested_collision_properties(
                f"/World/envs/env_0/{self.name}_{i}/frame",
                contact_offset=0.02,
            )

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

    def initialize(self, prim_paths_expr: str = None):
        super().initialize(prim_paths_expr)
        self.drone.initialize(f"/World/envs/env_.*/{self.name}_*/{self.drone.name}_*")

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        self.drone.apply_action(actions)
    
    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

    def _create_frame(
        self, 
        prim_path: str,
        to_prims: Sequence[str]=None,
        enable_collision: bool=False
    ):
        frame_prim = create_frame(
            prim_path,
            self.arm_angles,
            self.arm_lengths,
            to_prims,
            joint_damping=self.joint_damping,
            enable_collision=enable_collision,
            exclude_from_articulation=False
        )
        return frame_prim

