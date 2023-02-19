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
    import omni_drones.utils.kit as kit_utils
    from omni_drones.robots import drone, RobotBase, RobotCfg

    sim = SimulationContext(
        stage_units_in_meters=1.0, 
        physics_dt=0.005, rendering_dt=0.005, 
        sim_params=cfg.sim,
        backend="torch", device="cuda"
    )
    
    drones: Dict[str, RobotBase] = {}
    n = 3
    for i, model in enumerate([
        "Crazyflie", 
        "Firefly", 
        "Hummingbird",
        # "Neo11", 
        # "Omav"
    ]):
        cfg = RobotCfg()
        drones[model] = getattr(drone, model)(cfg=cfg)
        translation = torch.zeros(n, 3)
        translation[:, 0] = i 
        translation[:, 1] = torch.arange(n)
        translation[:, 2] = 0.5
        drones[model].spawn(n, translation=translation)

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
    sim.reset()

    for _drone in drones.values():
        _drone.initialize()

    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=not cfg.headless)
            continue
        for _drone in drones.values():
            actions = _drone.action_spec.zero((_drone._count,))
            actions.fill_(0.)
            _drone.apply_action(actions)
        sim.step()

    simulation_app.close()

if __name__ == "__main__":
    main()