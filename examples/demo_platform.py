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

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )

    drone_model = "Firefly"
    n = 4
    translations = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )

    drone: MultirotorBase = MultirotorBase.REGISTRY[drone_model]()
    controller = drone.DEFAULT_CONTROLLER(
        dt=sim.get_physics_dt(), g=9.81, uav_params=drone.params
    ).to(sim.device)
    controller_state = TensorDict({}, n, device=sim.device)
    control_target = torch.zeros(n, 7, device=sim.device)
    control_target[:, 0:3] = (translations + torch.tensor([3.0, 0.0, 1.5])).to(
        cfg.sim.device
    )

    platform = prim_utils.create_prim(
        "/World/envs/env_0/platform", translation=(0, 0, 1)
    )
    drone.spawn(
        translations=translations,
        prim_paths=[f"/World/envs/env_0/platform/Firefly_{i}" for i in range(n)],
    )

    arm_angles = [torch.pi * 2 / n * i for i in range(n)]
    arm_lengths = [1.0 for _ in range(n)]
    scene_utils.create_frame(
        "/World/envs/env_0/platform/frame",
        arm_angles,
        arm_lengths,
        [f"/World/envs/env_0/platform/Firefly_{i}/base_link" for i in range(n)],
    )

    scene_utils.design_scene()
    sim.reset()
    sim.pause()

    drone.initialize("/World/envs/env_0/platform/Firefly_*")

    while simulation_app.is_running():
        if not sim.is_playing():
            sim.step()
            continue
        root_state = drone.get_state()[..., :13].squeeze(0)
        action, controller_state = functorch.vmap(controller)(
            root_state, control_target, controller_state
        )
        drone.apply_action(action)
        sim.step()

    simulation_app.close()


if __name__ == "__main__":
    main()
