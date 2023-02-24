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

    from omni_drones.robots import drone, RobotBase

    sim = SimulationContext(
        stage_units_in_meters=1.0, 
        physics_dt=0.005, rendering_dt=0.005, 
        sim_params=cfg.sim,
        backend="torch", device="cuda:0"
    )
    cloner = GridCloner(spacing=1.)
    cloner.define_base_env("/World/envs")
    prim_utils.define_prim("/World/envs/env_0")

    drones: Dict[str, RobotBase] = {}
    n = 2
    for i, model in enumerate([
        "Firefly", # "Hummingbird", "Neo11", "Omav"
    ]):
        drones[model] = getattr(drone, model)()
        translation = torch.zeros(n, 3)
        translation[:, 0] = i * 0.5
        translation[:, 1] = torch.arange(n) * 0.5
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

    prim_utils.create_prim(
        "/World/envs/env_0/target",
        "Sphere", attributes={"radius": 0.1, "primvars:displayColor": [(.5, 0.5, 0.5)]},
        translation=(0., 0., 0.5)
    )

    # VisualSphere("/World/envs/env_0/visual_sphere", radius=0.2, color=torch.tensor([0., 0., 1.]))
    # FixedSphere("/World/envs/env_0/fixed_sphere", radius=0.1, color=torch.tensor([0., 1., 0.]))
    # DynamicSphere("/World/envs/env_0/dynamic_sphere", radius=0.1, color=torch.tensor([1., 0., 0.]))
    
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

    for _drone in drones.values():
        _drone.initialize()
        pos, rot = _drone.get_world_poses()
        _drone.set_env_poses(pos, rot)
        
    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if sim.is_playing():
            sim.step()
        for _drone in drones.values():
            actions = _drone.action_spec.rand((cfg.env.num_envs, _drone._count,))
            _drone.apply_action(actions)

    simulation_app.close()

if __name__ == "__main__":
    main()