import logging
import os
import time

import hydra
import torch
import torch.nn as nn
import wandb
import torchopt
from functorch import vmap, grad_and_value
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.bspline import splev_torch, init_traj, get_ctps

from setproctitle import setproctitle
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from omni_drones.utils.torchrl.transforms import DepthImageNorm

from tqdm import tqdm
from functools import partial


def collision_cost_cylinders(wpts: torch.Tensor, cylinders: torch.Tensor, thres=0.5):
    cdist = torch.cdist(wpts, cylinders).min(dim=-1).values
    cost = torch.where(cdist < thres, thres - cdist, torch.zeros_like(cdist)).sum()
    return cost

class Planner(nn.Module):

    def __init__(self, n_ctps, k, device, opt_iters:int=60) -> None:
        super().__init__()
        self.opt = torchopt.adam(lr=0.1)
        self.n_ctps = n_ctps
        self.k = k
        self.init_traj = partial(init_traj, n_ctps=self.n_ctps, k=self.k)
        self.t = torch.linspace(0, n_ctps - k, 80, device=device)
        self.opt_iters = opt_iters

    def forward(self, start_state, end_pos, trees):
        start_pos = start_state[..., :3].reshape(-1, 3)
        start_linvel = start_state[..., 7:10].reshape(-1, 3)
        end_pos = end_pos.reshape(-1, 3)

        ctps, knots, traj, loss = vmap(self._plan)(start_pos, start_linvel, end_pos, trees)

        return ctps.unsqueeze(1), knots.unsqueeze(1)
    
    def _plan(self, start_pos, start_vel, end_pos, trees):
        ctps_init, knots = self.init_traj(
            start_pos, 
            end_pos,
            start_vel=start_vel, 
        )

        x = ctps_init[3:-3].clone()
        opt_state = self.opt.init(x)

        @grad_and_value
        def cost(x):
            ctps = get_ctps(ctps_init, x)
            traj = splev_torch(self.t, knots, ctps, k=self.k)
            cost_collision = collision_cost_cylinders(traj[..., :2], cylinders=trees)
            return cost_collision

        for i in range(self.opt_iters):
            grad, loss = cost(x)
            updates, opt_state = self.opt.update(grad, opt_state)
            x = x + updates

        ctps = get_ctps(ctps_init, x)
        traj = splev_torch(self.t, knots, ctps, k=self.k)
        return ctps, knots, traj, loss


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs import Forest
    from omni.isaac.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()

    base_env = Forest(cfg, headless=cfg.headless)

    controller = base_env.drone.DEFAULT_CONTROLLER(
        base_env.drone.dt, 9.81, base_env.drone.params
    ).to(base_env.device)

    transforms = Compose(
        InitTracker(), 
        DepthImageNorm([("drone.obs", "distance_to_camera")], 0, 24)
    )
    env = TransformedEnv(base_env, transforms)

    n = 16
    k = 4
    planner = TensorDictModule(
        Planner(n, k, base_env.device),
        in_keys=[("drone.obs", "state"), ("info", "target_pos"), ("info", "trees_pos")],
        out_keys=["ctps", "knots"]
    ) 

    def policy(tensordict: TensorDict):
        if "ctps" not in tensordict.keys():
            tensordict["ctps"] = torch.zeros(tensordict.shape[0], base_env.drone.n, n, 3, device=tensordict.device)
            tensordict["knots"] = torch.zeros(tensordict.shape[0], base_env.drone.n, n + k + 1, device=tensordict.device)
            tensordict.update(env.action_spec.zero())
        is_init = tensordict.get("is_init").squeeze(-1).nonzero().squeeze(-1)
        if len(is_init):
            planner_ouput = planner(tensordict[is_init])
            tensordict.update_at_(planner_ouput, is_init)

        knots = tensordict["knots"].squeeze(1)
        ctps = tensordict["ctps"].squeeze(1)
        t = tensordict["info"].get("time").expand(-1, base_env.drone.n).clamp_max(knots[:, [-1]])

        drone_state = tensordict[("drone.obs", "state")][..., :13].squeeze(1)
        # compute control target by evaluating the spline
        control_target_pos = vmap(splev_torch)(t, knots, ctps, k=k).squeeze(1)
        control_target_vel = vmap(splev_torch)(t, knots, ctps, k=k, der=1).squeeze(1)
        control_target_yaw = torch.atan2(control_target_vel[..., 1], control_target_vel[..., 0]).unsqueeze(-1)
        control_target_yaw.zero_()
        control_target = torch.cat([control_target_pos, control_target_vel, control_target_yaw], dim=-1)
        controller_state = tensordict.get("controller_state", TensorDict({}, tensordict.shape))
        cmds, controller_state = vmap(controller)(drone_state, control_target, controller_state)
        tensordict["control_target"] = control_target
        tensordict["drone.action"] = cmds.unsqueeze(1)
        tensordict["controller_state"] = controller_state

        return tensordict

    base_env.enable_render(True)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=base_env.num_envs * base_env.max_episode_length,
        total_frames=base_env.num_envs * base_env.max_episode_length * 2,
        device=cfg.sim.device,
        storing_device="cpu",
        return_same_td=True,
    )

    if not os.path.isdir("trajectories"):
        os.mkdir("trajectories")
    
    pbar = tqdm(collector)
    for i, batch in enumerate(pbar):
        torch.save(
            batch.select(
                ("drone.obs", "distance_to_camera"),
                ("drone.obs", "state"),
                "control_target",
                "drone.action",
            ), 
            f"trajectories/{i}.pth"
        )
        pbar.set_postfix(
            {
                "rollout_fps": collector._fps,
                "frames": collector._frames,
            }
        )

    from torchvision.io import write_video

    print(batch)
    rgb = batch.get(("drone.obs", "rgb"), None) # [env, T, agent, H, W, C]
    if rgb is not None:
        for env_id, video_array in tqdm(enumerate(rgb[:, :, 0].unbind(0))):
            write_video(f"rgb_{env_id}.mp4", video_array[:, :3].permute(0, 2, 3, 1), fps=50)
    
    # depth = batch.get(("drone.obs", "distance_to_camera"), None)
    # if depth is not None:
    #     for env_id, video_array in tqdm(enumerate(depth[:, :, 0].unbind(0))):
    #         video_array =  video_array.expand(*video_array.shape[:-1], 3)
    #         write_video(f"depth_{env_id}.mp4", video_array, fps=50)

    simulation_app.close()


if __name__ == "__main__":
    main()
