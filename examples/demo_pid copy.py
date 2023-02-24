import torch, functorch
import hydra
import os

from typing import Dict, Optional
from omegaconf import OmegaConf
from tensordict import TensorDict
from omni_drones import CONFIG_PATH, init_simulation_app

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni.isaac.core.simulation_context import SimulationContext
    import omni.isaac.core.utils.prims as prim_utils
    import omni_drones.utils.kit as kit_utils
    from omni_drones.robots.drone import MultirotorBase, Crazyflie, Firefly, Hummingbird
    from omni_drones.robots import RobotCfg
    from omni_drones.controllers.cf2x_pid import PIDController, DSLPIDControl

    sim = SimulationContext(
        stage_units_in_meters=1.0, 
        physics_dt=0.005, rendering_dt=0.005, 
        sim_params=cfg.sim,
        backend="torch", device="cuda"
    )
    n = 2

    cfg = RobotCfg()
    drones = []
    cf2x = Crazyflie(cfg=cfg)

    translation = torch.zeros(n, 3)
    translation[:, 1] = torch.arange(n)
    translation[:, 2] = 0.5
    cf2x.spawn(n, translation=translation)

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
    cf2x.initialize()
    init_poses = cf2x.get_env_poses()
    init_vels = cf2x.get_velocities()
    controller = PIDController(
        dt=sim.get_physics_dt(), g=9.81, mass=cf2x.mass
    ).to(sim.device)

    # init_poses = firefly.get_env_poses()
    # init_vels = firefly.get_velocities()
    # controller = PositionController(
    #     g=9.81, mass=firefly.mass, uav_config=firefly.params
    # ).to(sim.device)

    controller_state = TensorDict({}, n, device=sim.device)
    control_target = torch.zeros(n, 7, device=sim.device)
    control_target[:, 0] = 0.3
    control_target[:, 1] = torch.arange(n, device=sim.device) + 0.2
    control_target[:, 2] = 1.0

    action = cf2x.action_spec.zero((n,))
    # max_thrust = cf2x.mass * cf2x.thrust_to_weight * 9.81 / cf2x.num_rotors
    # max_thrust = firefly.mass * firefly.thrust_to_weight * 9.81 / firefly.num_rotors
    # action = firefly.action_spec.zero((n,))

    step = 0
    
    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            continue
        root_state = cf2x.get_state()[..., :13].squeeze(0)
        # root_state = firefly.get_state()[..., :13].squeeze(0)
        # action, controller_state = functorch.vmap(controller)(
        #     root_state, control_target, controller_state)
        action[0], controller_state[0] = controller(
            root_state[0], control_target[0], controller_state[0])
        cf2x.apply_action(action)
        # firefly.apply_action(action)
        sim.step()
        step += 1
        if step == 800:
            cf2x.set_env_poses(*init_poses)
            cf2x.set_velocities(init_vels)
            # firefly.set_env_poses(*init_poses)
            # firefly.set_velocities(init_vels)
            controller_state.zero_()
            step = 0
            sim.step()

    simulation_app.close()

EPS = 1e-7
def normalize(x: torch.Tensor, norm: Optional[torch.Tensor] = None):
    if norm is None:
        norm = torch.norm(x, dim=-1, keepdim=True)
    return x / (norm + EPS), norm

if __name__ == "__main__":
    main()