import torch
import hydra
import os

from typing import Dict
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni.isaac.core.simulation_context import SimulationContext
    import omni.isaac.core.utils.prims as prim_utils
    import omni.isaac.core.utils.stage as stage_utils
    import omni.physx.scripts.utils as script_utils
    import omni_drones.utils.kit as kit_utils
    import omni_drones.utils.math as math_utils
    from omni_drones.robots.drone import MultirotorBase
    from helpers import design_scene

    sim = SimulationContext(
        stage_units_in_meters=1.0, 
        physics_dt=0.005, rendering_dt=0.005, 
        sim_params=cfg.sim,
        backend="torch", 
        device=cfg.sim.device
    )
    
    drone_model = "Firefly"
    n = 4
    translation = torch.tensor([
        [-1., 0., 1.0],
        [0., 1., 1.0],
        [1., 0., 1.0],
        [0., -1., 1.0]
    ])
    drone = MultirotorBase.REGISTRY[drone_model]()
    
    drone.spawn(
        n=n, translation=translation,
    )

    stage = stage_utils.get_current_stage()
    orientation_euler = torch.zeros(n, 3)
    orientation_euler[:, 2] = torch.pi * 2 * torch.linspace(0, 1, n+1)[:-1]
    orientation = math_utils.euler_to_quaternion(orientation_euler)

    base_translation = torch.tensor([0., 0., 1.])
    platform = prim_utils.create_prim(
        "/World/envs/env_0/Platform",
        translation=base_translation
    )
    base_link = prim_utils.create_prim(
        "/World/envs/env_0/Platform/body", prim_type="Sphere", 
        attributes={"radius": 0.05}
    )
    for i in range(n):
        arm_prim = prim_utils.create_prim(
            f"/World/envs/env_0/Platform/arm_{i}", prim_type="Capsule", 
            translation=(translation[i]-base_translation)/2,
            attributes={"radius": 0.02, "axis": "X", "height": 0.7}
        )
        joint = script_utils.createJoint(stage, "Fixed", base_link, arm_prim)
        script_utils.setRigidBody(arm_prim, "convexHull", kinematic=False)
    
    # drone_prim = prim_utils.get_prim_at_path(f"/World/envs/env_0/Firefly_0/base_link")
    # joint_arm_drone = script_utils.createJoint(stage, "Revolute", drone_prim, arm_prim)
    # joint_arm_drone.GetAttribute("physics:excludeFromArticulation").Set(True)

    # drone_prim = prim_utils.get_prim_at_path(f"/World/envs/env_0/Firefly_2/base_link")
    # joint_arm_drone = script_utils.createJoint(stage, "Revolute", drone_prim, arm_prim)
    # joint_arm_drone.GetAttribute("physics:excludeFromArticulation").Set(True)
    
    # drone_prim = prim_utils.get_prim_at_path(f"/World/envs/env_0/Firefly_1/base_link")
    # joint_arm_drone = script_utils.createJoint(stage, "Revolute", drone_prim, arm_prim)
    # joint_arm_drone.GetAttribute("physics:excludeFromArticulation").Set(True)

    # drone_prim = prim_utils.get_prim_at_path(f"/World/envs/env_0/Firefly_3/base_link")
    # joint_arm_drone = script_utils.createJoint(stage, "Revolute", drone_prim, arm_prim)
    # joint_arm_drone.GetAttribute("physics:excludeFromArticulation").Set(True)
    script_utils.setRigidBody(base_link, "convexHull", kinematic=False)

    design_scene()
    sim.reset()

    drone.initialize()

    while simulation_app.is_running():
        if sim.is_stopped():
            break
        sim.step()

    simulation_app.close()

if __name__ == "__main__":
    main()

"""


from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.prims.xform_prim_view import XFormPrimView

path = "/World/Xform_*"
try:
    a = ArticulationView(
        path
    )
    a.initialize()
    print("articulation:", a.get_world_poses()[0])
except Exception as e:
    print(e)

try:
    b = XFormPrimView(
        path
    )
    print("body:", b.get_world_poses()[0])
except Exception as e:
    print(e)

"""