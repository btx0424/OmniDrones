from typing import Sequence, Union, Optional

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.physx.scripts.utils as script_utils
import torch

from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform.rotation import Rotation


def create_frame(
    prim_path: str,
    arm_angles: Sequence[float],
    arm_lengths: Sequence[float],
    to_prim_paths: Optional[Sequence[str]]=None,
    joint_damping: float = 0.005,
    color: Sequence[float] = (0.5, 0.5, 0.2),
    enable_collision: bool = False,
):
    if not len(arm_angles) == len(arm_lengths):
        raise ValueError
    if isinstance(arm_angles, torch.Tensor):
        arm_angles = arm_angles.tolist()
    if isinstance(arm_lengths, torch.Tensor):
        arm_lengths = arm_lengths.tolist()

    stage = stage_utils.get_current_stage()
    prim_xform = prim_utils.define_prim(prim_path)
    
    arms = []
    if to_prim_paths is None:
        to_prim_paths = [None for _ in range(len(arm_angles))]
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

        arms.append(capsuleGeom.GetPrim())

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
            joint.GetAttribute("physics:excludeFromArticulation").Set(True)

            UsdPhysics.DriveAPI.Apply(joint, "rotX")
            UsdPhysics.DriveAPI.Apply(joint, "rotY")
            joint.GetAttribute("drive:rotX:physics:damping").Set(joint_damping)
            joint.GetAttribute("drive:rotY:physics:damping").Set(joint_damping)

    script_utils.setRigidBody(prim_xform, "convexHull", False)

    massAPI = UsdPhysics.MassAPI.Apply(prim_xform)
    massAPI.CreateMassAttr().Set(0.2)

    for arm in arms:
        arm.GetAttribute("physics:collisionEnabled").Set(enable_collision)
    return prim_xform

