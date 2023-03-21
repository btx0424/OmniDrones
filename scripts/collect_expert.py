import logging
import os
import time

import hydra
import torch
import wandb
import torchopt
from functorch import vmap, grad_and_value
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.learning.collectors import SyncDataCollector
from omni_drones.utils.envs.transforms import LogOnEpisode
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.math import quaternion_to_euler

from setproctitle import setproctitle
from tensordict import TensorDict
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs.transforms import TransformedEnv, InitTracker

from tqdm import tqdm


def collision_cost_cylinders(wpts: torch.Tensor, cylinders: torch.Tensor, thres=0.5):
    cdist = torch.cdist(wpts, cylinders).min(dim=-1).values
    cost = torch.where(cdist < thres, thres - cdist, torch.zeros_like(cdist)).sum()
    return cost


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    # run = init_wandb(cfg)
    # setproctitle(run.name)
    # print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs import Forest
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg
    from omni_drones.utils.bspline import splev_torch, init_traj, get_ctps

    from omni.isaac.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()

    base_env = Forest(cfg, headless=cfg.headless)

    controller = base_env.drone.DEFAULT_CONTROLLER(
        base_env.drone.dt, 9.81, base_env.drone.params
    ).to(base_env.device)

    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(320, 240),
        data_types=["rgb", "distance_to_camera"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.3, 1.0e5),
        ),
    )
    camera = Camera(camera_cfg)
    camera.spawn(
        [f"/World/envs/env_0/Firefly_{i}/base_link/Camera" for i in range(base_env.drone.n)],
        targets=[(1., 0., 0.1)]
    )
    camera.initialize("/World/envs/.*/Firefly_*/base_link/Camera")
    env = TransformedEnv(base_env, InitTracker())

    n = 12
    k = 4
    opt = torchopt.adam(lr=0.1)
    t = torch.linspace(0, n - k, 80, device=base_env.device)

    def plan(start_pos, start_vel, end_pos, trees):
        ctps_init, knots = init_traj(
            start_pos, 
            end_pos,
            start_vel=start_vel, 
            n_ctps=n, 
            k=k
        )

        x = ctps_init[3:-3].clone()
        opt_state = opt.init(x)

        @grad_and_value
        def cost(x):
            ctps = get_ctps(ctps_init, x)
            traj = splev_torch(t, knots, ctps, k=k)
            cost_collision = collision_cost_cylinders(traj[..., :2], cylinders=trees)
            return cost_collision

        for i in range(60):
            grad, loss = cost(x)
            updates, opt_state = opt.update(grad, opt_state)
            x = x + updates

        ctps = get_ctps(ctps_init, x)
        traj = splev_torch(t, knots, ctps, k=k)
        return ctps, knots, traj, loss

    def planner(tensordict: TensorDict):
        start_pos = tensordict.get("drone.obs")[..., :3].reshape(-1, 3)
        start_linvel = tensordict.get("drone.obs")[..., 7:10].reshape(-1, 3)
        end_pos = tensordict["info"].get("target_pos").reshape(-1, 3)
        trees = tensordict["info"].get("trees_pos")

        ctps, knots, traj, loss = vmap(plan)(start_pos, start_linvel, end_pos, trees)

        point_list_0 = traj[0, :-1] + base_env.envs_positions[0]
        point_list_1 = traj[0, 1:] + base_env.envs_positions[0]
        colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
        sizes = [1 for _ in range(len(point_list_0))]

        draw.clear_lines()
        draw.draw_lines(point_list_0.tolist(), point_list_1.tolist(), colors, sizes)

        return ctps.unsqueeze(1), knots.unsqueeze(1)

    frames = []

    def policy(tensordict: TensorDict):
        if "ctps" not in tensordict.keys():
            tensordict["ctps"] = torch.zeros(tensordict.shape[0], base_env.drone.n, n, 3, device=tensordict.device)
            tensordict["knots"] = torch.zeros(tensordict.shape[0], base_env.drone.n, n + k + 1, device=tensordict.device)
            tensordict.update(env.action_spec.zero())
        is_init = tensordict.get("is_init")
        if is_init.any():
            ctps, knots = planner(tensordict[is_init])
            tensordict.set_at_("ctps", ctps, is_init)
            tensordict.set_at_("knots", knots, is_init)

        t = tensordict["info"].get("time").expand(-1, base_env.drone.n)
        knots = tensordict["knots"].squeeze(1)
        ctps = tensordict["ctps"].squeeze(1)

        drone_state = tensordict.get("drone.obs")[..., :13].squeeze(1)
        # compute control target by evaluating the spline
        controller_target_pos = vmap(splev_torch)(t, knots, ctps, k=k).squeeze(1)
        controller_target_vel = vmap(splev_torch)(t, knots, ctps, k=k, der=1).squeeze(1)
        controller_target_yaw = torch.atan2(controller_target_vel[..., 1], controller_target_vel[..., 0]).unsqueeze(-1)
        controller_target_yaw.zero_()
        controller_target = torch.cat([controller_target_pos, controller_target_vel, controller_target_yaw], dim=-1)
        controller_state = tensordict.get("controller_state", TensorDict({}, tensordict.shape))
        cmds, controller_state = vmap(controller)(drone_state, controller_target, controller_state)
        tensordict["drone.action"] = cmds.unsqueeze(1)
        tensordict["controller_state"] = controller_state

        frames.append(camera.get_images().cpu())
        return tensordict

    base_env.enable_render(True)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=base_env.num_envs * base_env.max_episode_length,
        total_frames=base_env.num_envs * base_env.max_episode_length * 2,
        device=cfg.sim.device,
        return_same_td=True,
    )

    pbar = tqdm(collector)
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames}

        pbar.set_postfix(
            {
                "rollout_fps": collector._fps,
                "frames": collector._frames,
            }
        )
    
    from torchvision.io import write_video

    for k, v in torch.stack(frames).cpu().items():
        for i, vv in enumerate(v.unbind(1)):
            if vv.shape[-1] == 4:
                write_video(f"{k}_{i}.mp4", vv[..., :3], fps=50)
            elif vv.shape[-1] == 1:
                vv = -torch.nan_to_num(vv, 0).expand(*vv.shape[:-1], 3)
                write_video(f"{k}_{i}.mp4", vv[..., :3], fps=50)

    simulation_app.close()


if __name__ == "__main__":
    main()
