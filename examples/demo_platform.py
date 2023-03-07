import torch
import hydra
import os
import functorch

from tensordict import TensorDict
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
    import omni_drones.utils.scene as scene_utils
    from omni_drones.robots.drone import MultirotorBase

    sim = SimulationContext(
        stage_units_in_meters=1.0, 
        physics_dt=0.01, rendering_dt=0.01, 
        sim_params=cfg.sim,
        backend="torch", 
        device=cfg.sim.device
    )
    
    drone_model = "Firefly"
    n = 4
    translations = torch.tensor([
        [-1., 0., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., -1., 0.]
    ])

    drone: MultirotorBase = MultirotorBase.REGISTRY[drone_model]()
    controller = drone.default_controller(
        dt=sim.get_physics_dt(), g=9.81, uav_params=drone.params
    ).to(sim.device)
    controller_state = TensorDict({}, n, device=sim.device)
    control_target = torch.zeros(n, 7, device=sim.device)
    control_target[:, 0:3] = (
        torch.tensor(translations) + torch.tensor([3.0, 0., 0.7])
    ).to(cfg.sim.device)

    platform = prim_utils.create_prim(
        "/World/envs/env_0/platform",
        translation=(0, 0, 1)
    )
    drone.spawn(
        n=n, translations=translations,
        prim_paths=[f"/World/envs/env_0/platform/Firefly_{i}" for i in range(n)]
    )

    arm_angles = [torch.pi*2/n * i for i in range(n)]
    arm_lengths = [1. for _ in range(n)]
    scene_utils.create_frame(
        "/World/envs/env_0/platform/frame", 
        arm_angles, arm_lengths,
        [f"/World/envs/env_0/platform/Firefly_{i}/base_link" for i in range(n)]
    )

    scene_utils.design_scene()
    sim.reset()
    sim.pause()

    drone.initialize("/World/envs/env_0/platform/Firefly_*")

    while simulation_app.is_running():
        if not sim.is_playing(): 
            sim.step()
            continue
        sim.step()

    simulation_app.close()

if __name__ == "__main__":
    main()

"""
import torch
import omni_drones.utils.scene as scene_utils

n =4
arm_angles = [torch.pi/n * i for i in range(n)]
arm_lengths = [0.5 for _ in range(n)]
scene_utils.create_frame("/World/envs/env_0/frame", arm_angles, arm_lengths)


"""