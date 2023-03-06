from typing import Union

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.physx.scripts.utils as script_utils

from pxr import Usd, UsdPhysics

def create_rope(
    prim_path: str,
    from_prim: Union[str, Usd.Prim]=None,
    to_prim: Union[str, Usd.Prim]=None,
    segment_lenth=0.08, 
    n_negments: int=10,
    joint_damping: float=1,
    joint_stiffness: float=0.1
):

    if prim_utils.is_prim_path_valid(prim_path):
        raise ValueError(f"Prim at path {prim_path} already exists.")
    if isinstance(from_prim, str):
        from_prim = prim_utils.get_prim_at_path(from_prim)
    if isinstance(to_prim, str):
        to_prim = prim_utils.get_prim_at_path(to_prim)
    
    stage = stage_utils.get_current_stage()
    parent_prim = prim_utils.create_prim(
        prim_path=prim_path,
        translation=(0., 0., 0.),
    )
    seg_0 = prim_utils.create_prim(
        prim_path=f"{prim_path}/seg_0",
        prim_type="Capsule",
        translation=(0., 0., 0.),
        attributes={"radius": 0.01, "height": segment_lenth*0.7}
    )
    script_utils.setRigidBody(seg_0, "convexHull", False)
    seg_0.GetAttribute('physics:collisionEnabled').Set(False)
    prims = [seg_0]
    for i in range(1, n_negments):
        seg_i = prim_utils.create_prim(
            prim_path=f"{prim_path}/seg_{i}",
            prim_type="Capsule",
            translation=(0., 0., -i*segment_lenth/2),
            attributes={"radius": 0.01, "height": segment_lenth*0.7}
        )
        UsdPhysics.MassAPI.Apply(seg_i)
        seg_i.GetAttribute("physics:mass").Set(0.01)
        joint: Usd.Prim = script_utils.createJoint(stage, "D6", seg_i, prims[-1])
        joint.GetAttribute('physics:localPos0').Set((0., 0., segment_lenth/2))
        joint.GetAttribute('physics:localPos1').Set((0., 0., -segment_lenth/2))
        joint.GetAttribute('limit:rotX:physics:low').Set(-90)
        joint.GetAttribute('limit:rotX:physics:high').Set(90)
        joint.GetAttribute('limit:rotY:physics:low').Set(-90)
        joint.GetAttribute('limit:rotY:physics:high').Set(90)
        UsdPhysics.DriveAPI.Apply(joint, "rotX")
        UsdPhysics.DriveAPI.Apply(joint, "rotY")
        joint.GetAttribute('drive:rotX:physics:damping').Set(joint_damping)
        joint.GetAttribute('drive:rotX:physics:stiffness').Set(joint_stiffness)
        joint.GetAttribute('drive:rotY:physics:damping').Set(joint_damping)
        joint.GetAttribute('drive:rotY:physics:stiffness').Set(joint_stiffness)

        script_utils.setRigidBody(seg_i, "covexHull", False)
        seg_i.GetAttribute('physics:collisionEnabled').Set(False)
        
        prims.append(seg_i)
    
    if from_prim is not None:
        joint: Usd.Prim = script_utils.createJoint(stage, "D6", from_prim, seg_i)

    if to_prim is not None:
        joint: Usd.Prim = script_utils.createJoint(stage, "D6", seg_0, to_prim)
        joint.GetAttribute('physics:excludeFromArticulation').Set(True)
    
    return prims