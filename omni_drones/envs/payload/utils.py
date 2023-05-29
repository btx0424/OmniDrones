import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import omni_drones.utils.kit as kit_utils
from pxr import UsdPhysics
import omni.isaac.core.objects as objects


def attach_payload(
    drone_prim_path: str,
    bar_length: str,
    payload_radius: float=0.08,
    payload_mass: float=0.3
):
    bar = prim_utils.create_prim(
        prim_path=drone_prim_path + "/bar",
        prim_type="Capsule",
        translation=(0., 0., -bar_length / 2.),
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
    joint.GetAttribute("limit:rotX:physics:low").Set(-120)
    joint.GetAttribute("limit:rotX:physics:high").Set(120)
    joint.GetAttribute("limit:rotY:physics:low").Set(-120)
    joint.GetAttribute("limit:rotY:physics:high").Set(120)
    UsdPhysics.DriveAPI.Apply(joint, "rotX")
    UsdPhysics.DriveAPI.Apply(joint, "rotY")
    joint.GetAttribute("drive:rotX:physics:damping").Set(0.0001)
    joint.GetAttribute("drive:rotY:physics:damping").Set(0.0001)
    # joint.GetAttribute('physics:excludeFromArticulation').Set(True)

    payload = objects.DynamicSphere(
        prim_path=drone_prim_path + "/payload",
        translation=(0., 0., -bar_length),
        radius=payload_radius,
        mass=payload_mass
    )
    joint = script_utils.createJoint(stage, "Fixed", bar, payload.prim)
    kit_utils.set_collision_properties(
        drone_prim_path + "/bar", contact_offset=0.02, rest_offset=0
    )
    kit_utils.set_collision_properties(
        drone_prim_path + "/payload", contact_offset=0.02, rest_offset=0
    )