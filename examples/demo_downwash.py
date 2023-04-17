import os

from typing import Dict, Optional
import torch
from functorch import vmap

import hydra
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app
from tensordict import TensorDict


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni.isaac.core.utils.prims as prim_utils
    import omni_drones.utils.kit as kit_utils
    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.controllers import LeePositionController
    from omni_drones.robots import RobotCfg
    from omni_drones.robots.drone import Crazyflie, Firefly, Hummingbird, MultirotorBase
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )
    n = 3

    drone = Firefly(cfg=RobotCfg())

    translations = torch.zeros(n, 3)
    translations[:, 1] = torch.arange(n) * 0.5
    translations[:, 2] = torch.arange(n) + 1.0
    drone.spawn(translations=translations)

    scene_utils.design_scene()

    sim.reset()
    drone.initialize()

    init_poses = drone.get_world_poses(clone=True)
    init_vels = drone.get_velocities(clone=True)
    controller = drone.DEFAULT_CONTROLLER(
        dt=sim.get_physics_dt(), g=9.81, uav_params=drone.params
    ).to(sim.device)

    controller_state = TensorDict({}, n, device=sim.device)
    control_target = torch.zeros(n, 7, device=sim.device)
    control_target[:, 0] = 0
    control_target[:, 1] = translations[:, 1]
    control_target[:, 2] = translations[:, 2]
    action = drone.action_spec.zero((n,))
    
    
    from tqdm import tqdm
    t = tqdm(range(2000))
    for i in t:
        if sim.is_stopped():
            break
        if not sim.is_playing():
            continue
        root_state = drone.get_state(env=False)[..., :13].squeeze(0)
        downwash_forces = MultirotorBase.downwash(
            root_state[:, :3],
            root_state[:, :3],
            drone.forces.squeeze(0).sum(1)
        )
        action, controller_state = vmap(controller)(
            root_state, control_target, controller_state
        )
        drone.apply_action(action)
        sim.step()

    simulation_app.close()


if __name__ == "__main__":
    main()
