import torch

import hydra
from omegaconf import OmegaConf
from omni_drones import init_simulation_app


@hydra.main(version_base=None, config_path=".", config_name="demo")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.utils.torch import euler_to_quaternion

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=cfg.sim.dt,
        rendering_dt=cfg.sim.dt,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )
    n = 4

    drone_model_cfg = cfg.drone_model
    drone, controller = MultirotorBase.make(
        drone_model_cfg.name, drone_model_cfg.controller, cfg.sim.device
    )

    translations = torch.zeros(n, 3)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 0.5
    drone.spawn(translations=translations)

    scene_utils.design_scene()

    sim.reset()
    drone.initialize()

    # let's fly a circular trajectory
    radius = 1.5
    omega = 1.
    phase = torch.linspace(0, 2, n+1, device=sim.device)[:n]

    def compute_ref(t):
        _t = phase * torch.pi + t * omega
        pos = torch.stack([
            torch.cos(_t) * radius,
            torch.sin(_t) * radius,
            torch.ones(n, device=sim.device) * 1.5
        ], dim=-1)
        vel_xy = torch.stack([
            -torch.sin(_t) * radius * omega,
            torch.cos(_t) * radius * omega,
        ], dim=-1)
        yaw = torch.atan2(vel_xy[:, 1], vel_xy[:, 0])
        return pos, yaw

    init_rpy = torch.zeros(n, 3, device=sim.device)
    init_pos, init_rpy[:, 2] = compute_ref(torch.tensor(0.0).to(sim.device))
    init_rot = euler_to_quaternion(init_rpy)
    init_vels = torch.zeros(n, 6, device=sim.device)

    def reset():
        drone._reset_idx(torch.tensor([0]))
        drone.set_world_poses(init_pos, init_rot)
        drone.set_velocities(init_vels)

    reset()
    drone_state = drone.get_state()[..., :13].squeeze(0)

    from tqdm import tqdm
    # the main loop
    for i in tqdm(range(1000)):
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue
        ref_pos, ref_yaw = compute_ref((i % 1000)*cfg.sim.dt)
        # compute rotor commands and apply
        action = controller.compute(drone_state, target_pos=ref_pos, target_yaw=ref_yaw)
        drone.apply_action(action)
        sim.step(render=True)

        if i % 1000 == 0:
            reset()
        drone_state = drone.get_state()[..., :13].squeeze(0)

    simulation_app.close()


if __name__ == "__main__":
    main()
