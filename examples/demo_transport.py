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

    import omni.isaac.core.objects as objects
    import omni.isaac.core.utils.prims as prim_utils
    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.robots.assembly.transportation_group import TransportationGroup
    from omni_drones.robots.drone import MultirotorBase

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )
    drone_model = "Firefly"
    drone: MultirotorBase = MultirotorBase.REGISTRY[drone_model]()
    controller = drone.DEFAULT_CONTROLLER(
        dt=sim.get_physics_dt(), g=9.81, uav_params=drone.params
    ).to(sim.device)
    controller_state = TensorDict({}, 4, device=sim.device)

    group = TransportationGroup(drone=drone)
    group.spawn(translations=[(0, 0, 1.5)])

    scene_utils.design_scene()
    sim.reset()
    group.initialize()
    init_poses = group.get_world_poses(True)
    init_joint_pos = group.get_joint_positions(True)
    init_drone_poses = drone.get_world_poses(True)

    print(init_drone_poses[0])

    control_target = torch.zeros(4, 7, device=sim.device)
    control_target[:, 0:3] = (
        torch.tensor(
            [
                [0.5, 0.5, 1.5],
                [0.5, -0.5, 1.5],
                [-0.5, -0.5, 1.5],
                [-0.5, 0.5, 1.5],
            ]
        )
        + torch.tensor([3.0, 0.0, 0.7])
    ).to(sim.device)

    step = 0
    while simulation_app.is_running():
        if not sim.is_playing():
            sim.step()
            continue
        root_state = drone.get_state(False)[..., :13].squeeze(0)
        action, controller_state = functorch.vmap(controller)(
            root_state, control_target, controller_state
        )
        # action = drone.action_spec.rand((4,))
        drone.apply_action(action)
        sim.step()
        step += 1
        if step % 500 == 0:
            group.set_world_poses(*init_poses)
            group.set_joint_positions(init_joint_pos)

            sim.step()
            print(drone.get_world_poses(True)[0])

    simulation_app.close()


if __name__ == "__main__":
    main()

"""

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

a = ArticulationView("/World/envs/env_0/TransportationGroup_0")
a.initialize()

d = RigidPrimView("/World/envs/env_0/TransportationGroup_0/unit_*/firefly/base_link")
d.initialize()

"""
