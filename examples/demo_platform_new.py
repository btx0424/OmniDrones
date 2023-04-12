import os

import functorch
import hydra
import torch
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app

from tensordict import TensorDict


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni.isaac.core.utils.prims as prim_utils
    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.envs.platform.utils import OveractuatedPlatform

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )

    drone_model = "Hummingbird"
    n = 4

    drone: MultirotorBase = MultirotorBase.REGISTRY[drone_model]()
    platform = OveractuatedPlatform(drone=drone)

    platform.spawn(
        [(0., 0., 2.)],
        arm_lengths=[0.85],
        enable_collision=True,
    )

    scene_utils.design_scene()
    sim.reset()
    sim.pause()

    platform.initialize()

    while simulation_app.is_running():
        if not sim.is_playing():
            sim.step()
            continue
        print(platform.get_world_poses())
        action = drone.action_spec.rand((4,))
        action[:] = -1
        drone.apply_action(action)
        sim.step()

    simulation_app.close()


if __name__ == "__main__":
    main()
