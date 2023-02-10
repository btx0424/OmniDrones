import torch
import hydra
import os

from typing import Dict
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp
from omni_drones import CONFIG_PATH

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    # launch the simulator
    config = {"headless": cfg.headless}
    # load cheaper kit config in headless
    if cfg.headless:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    else:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    # launch the simulator
    simulation_app = SimulationApp(config, experience=app_experience)

    from omni.isaac.core.simulation_context import SimulationContext
    import omni.isaac.core.utils.prims as prim_utils
    import omni.isaac.orbit.utils.kit as kit_utils
    from omni_drones.robots import drone, RobotBase

    sim = SimulationContext(
        stage_units_in_meters=1.0, 
        physics_dt=0.005, rendering_dt=0.005, 
        backend="torch", device="cuda"
    )
    
    drones: Dict[str, RobotBase] = {}
        
    for i, model in enumerate([
        "Crazyflie", # "Firefly", # "Hummingbird", "Neo11", "Omav"
    ]):
        drones[model] = getattr(drone, model)()
        drones[model].spawn()

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
        if sim.is_playing():
            sim.step()
        for _drone in drones.values():
            _drone.apply_action(_drone.action_spec.rand((_drone.count,)))

    simulation_app.close()

if __name__ == "__main__":
    main()