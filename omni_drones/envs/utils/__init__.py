from typing import Dict, Any, Sequence, Union
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
# import omni.kit.commands
import omni.usd.commands
from pxr import UsdGeom, Usd, UsdPhysics, Gf

import torch
import omni_drones.utils.kit as kit_utils

def create_obstacle(
    prim_path: str,
    prim_type: str,
    translation: Sequence[float],
    attributes: Dict,
):
    prim = prim_utils.create_prim(
        prim_path=prim_path,
        prim_type=prim_type,
        translation=translation,
        attributes=attributes
    )
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    prim.GetAttribute("physics:kinematicEnabled").Set(True)
    kit_utils.set_collision_properties(
        prim_path, contact_offset=0.02, rest_offset=0
    )

    return prim

DEFAULT_JOINT_ATTRIBUTES = {
    "limit:rotX:physics:low": -120,
    "limit:rotX:physics:high": 120, 
    "limit:rotY:physics:low": -120,
    "limit:rotY:physics:high": 120,
    "drive:rotX:physics:damping": 2e-6,
    "drive:rotY:physics:damping": 2e-6
}

def create_bar(
    prim_path: str,
    length: float,
    from_prim: Union[str, Usd.Prim]=None,
    to_prim: Union[str, Usd.Prim]=None,
    joint_from_attributes=None,
    joint_to_attributes=None,
):

    bar = prim_utils.create_prim(prim_path)
    seg_0 = prim_utils.create_prim(
        f"{prim_path}/seg_0", 
        "Capsule",
        translation=(0., 0., -length/2),
        attributes={"radius": 0.01, "height": length}
    )
    seg_1 = prim_utils.create_prim(
        f"{prim_path}/seg_1", 
        "Capsule", 
        translation=(0., 0., -length/2),
        attributes={"radius": 0.01, "height": length}
    )
    for seg in [seg_0, seg_1]:
        UsdPhysics.RigidBodyAPI.Apply(seg)
        UsdPhysics.CollisionAPI.Apply(seg)
        massAPI = UsdPhysics.MassAPI.Apply(seg)
        massAPI.CreateMassAttr().Set(0.001)
        
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "Prismatic", seg_0, seg_1)
    UsdPhysics.DriveAPI.Apply(joint, "linear")
    
    joint.GetAttribute("physics:axis").Set("Z")
    joint.GetAttribute("physics:upperLimit").Set(1.)
    joint.GetAttribute("physics:lowerLimit").Set(-1.)
    joint.GetAttribute("drive:linear:physics:damping").Set(10.0)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(1000.0)
    
    def setup_joint(joint, attributes):
        drives = set([key.split(":")[1] for key in attributes.keys() if key.startswith("drive")])
        for drive in drives:
            UsdPhysics.DriveAPI.Apply(joint, drive)
        for k, v in attributes.items():
            joint.GetAttribute(k).Set(v)

    if from_prim is not None:
        if isinstance(from_prim, str):
            from_prim = prim_utils.get_prim_at_path(from_prim)
        joint_from = script_utils.createJoint(stage, "D6", seg_0, from_prim)
        if joint_from_attributes is None:
            joint_from_attributes = DEFAULT_JOINT_ATTRIBUTES
        setup_joint(joint_from, joint_from_attributes)
        # omni.usd.commands.MovePrimCommand(
        #     path_from=from_prim.GetPath().pathString + "/D6Joint",
        #     path_to=prim_path + "/joint_from",
        #     keep_world_transform=False,
        #     time_code=Usd.TimeCode.Default(),
        #     destructive=False
        # ).do()
    
    if to_prim is not None:
        if isinstance(to_prim, str):
            to_prim = prim_utils.get_prim_at_path(to_prim)
        joint_to = script_utils.createJoint(stage, "Fixed", seg_0, to_prim)
        if joint_to_attributes is None:
            joint_to_attributes = DEFAULT_JOINT_ATTRIBUTES
        setup_joint(joint_to, joint_to_attributes)
        # omni.usd.commands.MovePrimCommand(
        #     path_from=to_prim.GetPath().pathString + "/D6Joint",
        #     path_to=prim_path + "/joint_to",
        #     keep_world_transform=False,
        #     time_code=Usd.TimeCode.Default(),
        #     destructive=False
        # ).do()
    
    return bar
        

def lemniscate(t, c):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    sin2p1 = torch.square(sin_t) + 1

    x = torch.stack([
        cos_t, sin_t * cos_t, c * sin_t
    ], dim=-1) / sin2p1.unsqueeze(-1)

    return x

def scale_time(t, a: float=1.0):
    return t / (1 + 1/(a*torch.abs(t)))


class TimeEncoding:
    def __init__(self, max_t):
        ...
    
    def encode(self, t):
        ...
