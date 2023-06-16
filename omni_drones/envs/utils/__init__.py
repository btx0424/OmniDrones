from typing import Dict, Any, Sequence
import omni.isaac.core.utils.prims as prim_utils
import omni_drones.utils.kit as kit_utils
from pxr import UsdPhysics
import torch

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

