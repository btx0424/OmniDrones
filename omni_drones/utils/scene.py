import torch
from typing import Union, Sequence
from scipy.spatial.transform.rotation import Rotation

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.physx.scripts.utils as script_utils
import omni_drones.utils.kit as kit_utils

from pxr import Usd, UsdPhysics, UsdGeom, Gf, PhysxSchema

def design_scene():
    kit_utils.create_ground_plane(
        "/World/defaultGroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.8,
        improve_patch_friction=True,
    )
    # Lights-1
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
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )

def create_rope(
    xform_path: str = "/World/rope",
    translation =(0, 0, 0),
    from_prim: Union[str, Usd.Prim]=None,
    to_prim: Union[str, Usd.Prim]=None,
    num_links: int=24,
    link_length: float=0.06,
    rope_damping: float=10.,
    rope_stiffness: float=1.,
    color =(0.4, 0.2, 0.1),
    enable_collision: bool=False,
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
    joint_offset = link_length/2 - link_length/8

    links = []
    for i in range(num_links):
        link_path = f"{xform_path}/seg_{i}"
        location = (i * (link_length-link_length/4), 0, 0)
        
        capsuleGeom = UsdGeom.Capsule.Define(stage, link_path)
        capsuleGeom.CreateHeightAttr(link_length/2)
        capsuleGeom.CreateRadiusAttr(link_radius)
        capsuleGeom.CreateAxisAttr("X")
        capsuleGeom.AddTranslateOp().Set(location)
        capsuleGeom.AddOrientOp().Set(Gf.Quatf(1.0))
        capsuleGeom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        capsuleGeom.CreateDisplayColorAttr().Set([color])

        UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
        massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
        massAPI.CreateMassAttr().Set(0.01)
        if enable_collision:
            UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
            physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(capsuleGeom.GetPrim())
            physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
            physxCollisionAPI.CreateContactOffsetAttr().Set(0.02)

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
            joint: Usd.Prim = script_utils.createJoint(stage, "D6", links[-1], capsuleGeom.GetPrim())
            joint.GetAttribute('physics:localPos0').Set((joint_offset, 0., 0.))
            joint.GetAttribute('physics:localPos1').Set((-joint_offset, 0., 0.))
            joint.GetAttribute('limit:rotY:physics:low').Set(-110)
            joint.GetAttribute('limit:rotY:physics:high').Set(110)
            joint.GetAttribute('limit:rotZ:physics:low').Set(-110)
            joint.GetAttribute('limit:rotZ:physics:high').Set(110)
            UsdPhysics.DriveAPI.Apply(joint, "rotY")
            UsdPhysics.DriveAPI.Apply(joint, "rotZ")
            joint.GetAttribute('drive:rotY:physics:damping').Set(rope_damping)
            joint.GetAttribute('drive:rotY:physics:stiffness').Set(rope_stiffness)
            joint.GetAttribute('drive:rotZ:physics:damping').Set(rope_damping)
            joint.GetAttribute('drive:rotZ:physics:stiffness').Set(rope_stiffness)

        links.append(capsuleGeom.GetPrim())
    
    if from_prim is not None: 
        joint: Usd.Prim = script_utils.createJoint(stage, "Fixed", from_prim, links[-1])
        # joint.GetAttribute('physics:excludeFromArticulation').Set(True)

    if to_prim is not None:
        joint: Usd.Prim = script_utils.createJoint(stage, "Fixed", links[0], to_prim)
        # joint.GetAttribute('physics:excludeFromArticulation').Set(True)

    return links


def create_frame(
    prim_path: str,
    arm_angles: Sequence[float],
    arm_lengths: Sequence[float],
    to_prims: Sequence[str],
    joint_damping: float=0.005,
    color: Sequence[float]=(.1, .3, .1),
    enable_collision: bool=False,
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
    for i, (arm_angle, arm_length, to_prim) in enumerate(zip(arm_angles, arm_lengths, to_prims)):
        link_path = f"{prim_path}/arm_{i}"
        capsuleGeom = UsdGeom.Capsule.Define(stage, link_path)

        capsuleGeom.CreateHeightAttr(arm_length)
        capsuleGeom.CreateRadiusAttr(arm_length*0.02)
        capsuleGeom.CreateAxisAttr("X")

        r = Rotation.from_euler("z", arm_angle)
        location = Gf.Vec3f(*r.apply((arm_length/2, 0, 0)))
        orient = Gf.Quatf(*r.as_quat()[[3, 0, 1, 2]])
        capsuleGeom.AddTranslateOp().Set(location)
        capsuleGeom.AddOrientOp().Set(orient)
        capsuleGeom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        capsuleGeom.CreateDisplayColorAttr().Set([color])

        arms.append(capsuleGeom.GetPrim())

        to_prim = prim_utils.get_prim_at_path(to_prim)
        joint: Usd.Prim = script_utils.createJoint(stage, "D6", prim_xform, to_prim)
        rel_orient = joint.GetAttribute('physics:localRot0').Get()
        joint.GetAttribute('physics:localRot0').Set(orient)
        joint.GetAttribute('physics:localRot1').Set(rel_orient * orient)
        joint.GetAttribute('limit:rotX:physics:low').Set(-torch.inf)
        joint.GetAttribute('limit:rotX:physics:high').Set(torch.inf)
        joint.GetAttribute('limit:rotY:physics:low').Set(-torch.inf)
        joint.GetAttribute('limit:rotY:physics:high').Set(torch.inf)
        joint.GetAttribute('physics:excludeFromArticulation').Set(True)

        UsdPhysics.DriveAPI.Apply(joint, "rotX")
        UsdPhysics.DriveAPI.Apply(joint, "rotY")
        joint.GetAttribute('drive:rotX:physics:damping').Set(joint_damping)
        joint.GetAttribute('drive:rotY:physics:damping').Set(joint_damping)
        

    script_utils.setRigidBody(prim_xform, "convexHull", False)

    massAPI = UsdPhysics.MassAPI.Apply(prim_xform)
    massAPI.CreateMassAttr().Set(0.5)

    for arm in arms:
        arm.GetAttribute('physics:collisionEnabled').Set(enable_collision)
    
