from typing import Sequence, Union, Optional

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.physx.scripts.utils as script_utils
import torch

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform.rotation import Rotation

import omni_drones.utils.kit as kit_utils


def design_scene():
    kit_utils.create_ground_plane(
        "/World/defaultGroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.8,
        improve_patch_friction=True,
    )
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 1200.0, "color": (1.0, 1.0, 1.0)},
    )


def create_rope(
    xform_path: str = "/World/rope",
    translation=(0, 0, 0),
    from_prim: Union[str, Usd.Prim] = None,
    to_prim: Union[str, Usd.Prim] = None,
    num_links: int = 24,
    link_length: float = 0.06,
    rope_damping: float = 10.0,
    rope_stiffness: float = 1.0,
    color=(0.4, 0.2, 0.1),
    enable_collision: bool = False,
):
    if isinstance(from_prim, str):
        from_prim = prim_utils.get_prim_at_path(from_prim)
    if isinstance(to_prim, str):
        to_prim = prim_utils.get_prim_at_path(to_prim)
    if isinstance(translation, torch.Tensor):
        translation = translation.tolist()

    stage = stage_utils.get_current_stage()
    ropeXform = UsdGeom.Xform.Define(stage, xform_path)
    ropeXform.AddTranslateOp().Set(Gf.Vec3f(*translation))
    ropeXform.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))
    link_radius = 0.02
    joint_offset = link_length / 2 - link_length / 8

    links = []
    for i in range(num_links):
        link_path = f"{xform_path}/seg_{i}"
        location = (i * (link_length - link_length / 4), 0, 0)

        capsuleGeom = UsdGeom.Capsule.Define(stage, link_path)
        capsuleGeom.CreateHeightAttr(link_length / 2)
        capsuleGeom.CreateRadiusAttr(link_radius)
        capsuleGeom.CreateAxisAttr("X")
        capsuleGeom.AddTranslateOp().Set(location)
        capsuleGeom.AddOrientOp().Set(Gf.Quatf(1.0))
        capsuleGeom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        capsuleGeom.CreateDisplayColorAttr().Set([color])

        UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
        massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
        massAPI.CreateMassAttr().Set(0.01)

        UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(capsuleGeom.GetPrim())
        # physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
        # physxCollisionAPI.CreateContactOffsetAttr().Set(0.02)
        capsuleGeom.GetPrim().GetAttribute("physics:collisionEnabled")

        if len(links) > 0:
            # jointPath = f"{link_path}/joint_{i}"
            # joint = UsdPhysics.Joint.Define(stage, jointPath)
            # joint.CreateBody0Rel().SetTargets([links[-1].GetPath()])
            # joint.CreateBody1Rel().SetTargets([link_path])

            # joint.CreateLocalPos0Attr().Set(Gf.Vec3f(joint_offset, 0, 0))
            # joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
            # joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-joint_offset, 0, 0))
            # joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

            # # locked DOF (lock - low is greater than high)
            # d6Prim = joint.GetPrim()
            # limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transX")
            # limitAPI.CreateLowAttr(1.0)
            # limitAPI.CreateHighAttr(-1.0)
            # limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transY")
            # limitAPI.CreateLowAttr(1.0)
            # limitAPI.CreateHighAttr(-1.0)
            # limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transZ")
            # limitAPI.CreateLowAttr(1.0)
            # limitAPI.CreateHighAttr(-1.0)
            # limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "rotX")
            # limitAPI.CreateLowAttr(1.0)
            # limitAPI.CreateHighAttr(-1.0)

            # # Moving DOF:
            # dofs = ["rotY", "rotZ"]
            # for d in dofs:
            #     limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, d)
            #     limitAPI.CreateLowAttr(-110)
            #     limitAPI.CreateHighAttr(110)

            #     # joint drives for rope dynamics:
            #     driveAPI = UsdPhysics.DriveAPI.Apply(d6Prim, d)
            #     driveAPI.CreateTypeAttr("force")
            #     driveAPI.CreateDampingAttr(rope_damping)
            #     driveAPI.CreateStiffnessAttr(rope_stiffness)
            joint: Usd.Prim = script_utils.createJoint(
                stage, "D6", links[-1], capsuleGeom.GetPrim()
            )
            joint.GetAttribute("physics:localPos0").Set((joint_offset, 0.0, 0.0))
            joint.GetAttribute("physics:localPos1").Set((-joint_offset, 0.0, 0.0))
            joint.GetAttribute("limit:rotY:physics:low").Set(-110)
            joint.GetAttribute("limit:rotY:physics:high").Set(110)
            joint.GetAttribute("limit:rotZ:physics:low").Set(-110)
            joint.GetAttribute("limit:rotZ:physics:high").Set(110)
            UsdPhysics.DriveAPI.Apply(joint, "rotY")
            UsdPhysics.DriveAPI.Apply(joint, "rotZ")
            joint.GetAttribute("drive:rotY:physics:damping").Set(rope_damping)
            joint.GetAttribute("drive:rotY:physics:stiffness").Set(rope_stiffness)
            joint.GetAttribute("drive:rotZ:physics:damping").Set(rope_damping)
            joint.GetAttribute("drive:rotZ:physics:stiffness").Set(rope_stiffness)

        links.append(capsuleGeom.GetPrim())

    if from_prim is not None:
        joint: Usd.Prim = script_utils.createJoint(stage, "Fixed", from_prim, links[-1])
        # joint.GetAttribute('physics:excludeFromArticulation').Set(True)

    if to_prim is not None:
        joint: Usd.Prim = script_utils.createJoint(stage, "Fixed", links[0], to_prim)
        joint.GetAttribute("physics:excludeFromArticulation").Set(True)

    return links


def create_bar(
    prim_path: str,
    length: float,
    translation=(0, 0, 0),
    from_prim: str = None,
    to_prim: str = None,
    mass: float = 0.02,
    enable_collision=False,
    color=(0.4, 0.4, 0.2),
):
    if isinstance(from_prim, str):
        from_prim = prim_utils.get_prim_at_path(from_prim)
    if isinstance(to_prim, str):
        to_prim = prim_utils.get_prim_at_path(to_prim)
    if isinstance(translation, torch.Tensor):
        translation = translation.tolist()

    stage = stage_utils.get_current_stage()

    capsuleGeom = UsdGeom.Capsule.Define(stage, f"{prim_path}/Capsule")
    capsuleGeom.CreateHeightAttr(length)
    capsuleGeom.CreateRadiusAttr(0.012)
    capsuleGeom.CreateAxisAttr("Z")
    capsuleGeom.AddTranslateOp().Set(Gf.Vec3f(*translation))
    capsuleGeom.AddOrientOp().Set(Gf.Quatf(1.0))
    capsuleGeom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    capsuleGeom.CreateDisplayColorAttr().Set([color])

    UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
    massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
    massAPI.CreateMassAttr().Set(mass)

    UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
    prim: Usd.Prim = capsuleGeom.GetPrim()
    prim.GetAttribute("physics:collisionEnabled").Set(enable_collision)

    if from_prim is not None:
        sphere = prim_utils.create_prim(
            f"{prim_path}/Sphere",
            "Sphere",
            translation=(0, 0, -length),
            attributes={"radius": 0.02},
        )
        UsdPhysics.RigidBodyAPI.Apply(sphere)
        UsdPhysics.CollisionAPI.Apply(sphere)
        sphere.GetAttribute("physics:collisionEnabled").Set(False)

        script_utils.createJoint(stage, "Fixed", from_prim, sphere)
        joint: Usd.Prim = script_utils.createJoint(stage, "D6", prim, sphere)
        joint.GetAttribute("limit:rotX:physics:low").Set(-120)
        joint.GetAttribute("limit:rotX:physics:high").Set(120)
        joint.GetAttribute("limit:rotY:physics:low").Set(-120)
        joint.GetAttribute("limit:rotY:physics:high").Set(120)
        UsdPhysics.DriveAPI.Apply(joint, "rotX")
        UsdPhysics.DriveAPI.Apply(joint, "rotY")
        joint.GetAttribute("drive:rotX:physics:damping").Set(0.0002)
        joint.GetAttribute("drive:rotY:physics:damping").Set(0.0002)

    if to_prim is not None:
        joint: Usd.Prim = script_utils.createJoint(stage, "D6", prim, to_prim)
        joint.GetAttribute("limit:rotX:physics:low").Set(-120)
        joint.GetAttribute("limit:rotX:physics:high").Set(120)
        joint.GetAttribute("limit:rotY:physics:low").Set(-120)
        joint.GetAttribute("limit:rotY:physics:high").Set(120)
        UsdPhysics.DriveAPI.Apply(joint, "rotX")
        UsdPhysics.DriveAPI.Apply(joint, "rotY")
        joint.GetAttribute("drive:rotX:physics:damping").Set(0.0002)
        joint.GetAttribute("drive:rotY:physics:damping").Set(0.0002)

    return prim


