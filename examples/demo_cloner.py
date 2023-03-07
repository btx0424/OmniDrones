import torch
import hydra
import os

from typing import Dict
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni.isaac.core.simulation_context import SimulationContext
    from omni.isaac.cloner import GridCloner
    from omni.isaac.core.prims import GeometryPrim, RigidPrim
    from omni.isaac.core.objects import DynamicSphere, FixedSphere, VisualSphere
    import omni.isaac.core.utils.prims as prim_utils
    import omni_drones.utils.kit as kit_utils
    import omni_drones.utils.scene as scene_utils

    from omni_drones.robots.drone import MultirotorBase

    sim = SimulationContext(
        stage_units_in_meters=1.0, 
        physics_dt=0.005, rendering_dt=0.005, 
        sim_params=cfg.sim,
        backend="torch", 
        device=cfg.sim.device
    )
    
    cloner = GridCloner(spacing=1.)
    cloner.define_base_env("/World/envs")
    prim_utils.define_prim("/World/envs/env_0")

    drones: Dict[str, MultirotorBase] = {}
    n = 2
    for i, model in enumerate([
        "Firefly", 
        "Hummingbird", 
        "Neo11"
    ]):
        drones[model] = MultirotorBase.REGISTRY[model]()
        translations = torch.zeros(n, 3)
        translations[:, 0] = i * 0.5
        translations[:, 1] = torch.arange(n) * 0.5
        translations[:, 2] = 0.5
        drones[model].spawn(n, translations=translations)

    scene_utils.design_scene()

    prim_utils.create_prim(
        "/World/envs/env_0/target",
        "Sphere", attributes={"radius": 0.1, "primvars:displayColor": [(.5, 0.5, 0.5)]},
        translation=(0., 0., 0.5)
    )
    
    # How to clone environments and make them independent by filtering out collisions.
    global_prim_paths = ["/World/defaultGroundPlane"]
    envs_prim_paths= cloner.generate_paths("/World/envs/env", cfg.env.num_envs)
    envs_positions = cloner.clone(
        source_prim_path="/World/envs/env_0",
        prim_paths=envs_prim_paths,
        replicate_physics=False,
    )
    envs_positions = torch.tensor(envs_positions, dtype=torch.float32, device=sim.device)
    physics_scene_path = sim.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", prim_paths=envs_prim_paths, global_paths=global_prim_paths
    )
    
    RobotBase._envs_positions = envs_positions.unsqueeze(1)
    sim.reset()

    for drone in drones.values():
        drone.initialize()
        pos, rot = drone.get_world_poses()
        drone.set_env_poses(pos, rot)
        
    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if sim.is_playing():
            sim.step()
        for drone in drones.values():
            actions = drone.action_spec.rand((cfg.env.num_envs, drone._count,))
            drone.apply_action(actions)

    simulation_app.close()

if __name__ == "__main__":
    main()