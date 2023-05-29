import torch

import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import omni.isaac.core.objects as objects
from pxr import UsdPhysics

def create_pendulum(
    drone_prim_path: str,
    bar_length: str,
    payload_radius: float=0.06,
    payload_mass: float=0.3
):
    bar = prim_utils.create_prim(
        prim_path=drone_prim_path + "/bar",
        prim_type="Capsule",
        translation=(0., 0., bar_length / 2.),
        attributes={"radius": 0.01, "height": bar_length}
    )
    bar.GetAttribute('primvars:displayColor').Set([(0.8, 0.1, 0.1)])
    UsdPhysics.RigidBodyAPI.Apply(bar)
    UsdPhysics.CollisionAPI.Apply(bar)
    massAPI = UsdPhysics.MassAPI.Apply(bar)
    massAPI.CreateMassAttr().Set(0.02)

    base_link = prim_utils.get_prim_at_path(drone_prim_path + "/base_link")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "D6", bar, base_link)
    joint.GetAttribute("limit:rotX:physics:low").Set(-torch.inf)
    joint.GetAttribute("limit:rotX:physics:high").Set(torch.inf)
    joint.GetAttribute("limit:rotY:physics:low").Set(-torch.inf)
    joint.GetAttribute("limit:rotY:physics:high").Set(torch.inf)
    UsdPhysics.DriveAPI.Apply(joint, "rotX")
    UsdPhysics.DriveAPI.Apply(joint, "rotY")
    joint.GetAttribute("drive:rotX:physics:damping").Set(0.0001)
    joint.GetAttribute("drive:rotY:physics:damping").Set(0.0001)

    payload = objects.DynamicSphere(
        prim_path=drone_prim_path + "/payload",
        translation=(0., 0., bar_length),
        radius=payload_radius,
        mass=payload_mass
    )
    joint = script_utils.createJoint(stage, "Fixed", bar, payload.prim)