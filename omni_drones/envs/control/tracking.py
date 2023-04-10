from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion
import omni.isaac.core.utils.prims as prim_utils
import torch

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec
from omni.isaac.debug_draw import _debug_draw

class Tracking(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.future_traj_len = self.cfg.task.future_traj_len
        assert self.future_traj_len > 0
        self.traj_func = {"circle": circle, "lemniscate": lemniscate}[self.cfg.task.traj_func]
        
        drone_state_dim = self.drone.state_spec.shape[-1]
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            UnboundedContinuousTensorSpec(drone_state_dim + 3*(self.future_traj_len-1)).to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )
        self.init_pos_scale = torch.tensor([4.0, 4.0, 2.0], device=self.device)
        self.traj_scale = torch.ones(self.num_envs, 1, 1, device=self.device)
        self.target_pos = torch.zeros(self.num_envs, self.future_traj_len, 3, device=self.device)
        self.phase = 0

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        cfg = RobotCfg()
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.drone.spawn(translations=[(0.0, 0.0, 1.5)])
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        pos, rot = self.init_poses
        self.drone._reset_idx(env_ids)
        self.drone.set_world_poses(pos[env_ids], rot[env_ids], env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        self.traj_scale[env_ids] = torch.rand(len(env_ids), 1, 1, device=self.device) + 2

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()

        t = self.phase + (self.progress_buf.unsqueeze(1) + torch.arange(self.future_traj_len, device=self.device)) * self.dt
        self.target_pos[:] = vmap(self.traj_func)(t)
        self.target_pos[..., :2] *= self.traj_scale
        self.target_pos[..., 2] = 1.5

        obs = torch.cat([
            (self.target_pos - self.root_state[..., :3]).flatten(1).unsqueeze(1),
            self.root_state[..., 3:],
        ], dim=-1)

        if self._should_render(0):
            # visualize the trajectory
            self.draw.clear_lines()
            point_list_0 = (self.root_state[self.central_env_idx, 0, :3] + self.envs_positions[self.central_env_idx]).tolist()
            point_list_1 = (self.target_pos[self.central_env_idx, 0, :3] + self.envs_positions[self.central_env_idx]).tolist()
            self.draw.draw_lines([point_list_0], [point_list_1], [(1.0, 1.0, 1.0, 1.0)], [1])

            t = self.phase + torch.arange(self.max_episode_length, device=self.device) * self.dt
            traj_vis = self.traj_func(t)
            traj_vis[:, :2] *= self.traj_scale[self.central_env_idx]
            traj_vis[:, 2] = 1.5
            traj_vis = traj_vis + self.envs_positions[self.central_env_idx]
            point_list_0 = traj_vis[:-1].tolist()
            point_list_1 = traj_vis[1:].tolist()
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [1 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

        return TensorDict({"drone.obs": obs}, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot, vels, heading, up = self.root_state[..., :19].split([3, 4, 6, 3, 3], dim=-1)

        # pos reward
        rpos = self.target_pos[:, 0, :].unsqueeze(1) - pos
        target_dist = torch.norm(rpos, dim=-1)
        
        k = 2.
        pose_reward = 1.0 / (1.0 + torch.square(k * target_dist))
        
        # uprightness
        tiltage = torch.abs(1 - up[..., 2])
        up_reward = 1.0 / (1.0 + torch.square(tiltage))

        # effort
        effort_reward = 0.1 * torch.exp(-self.effort)

        # spin reward
        spin = torch.square(vels[..., -1])
        spin_reward = 1.0 / (1.0 + torch.square(spin))

        assert pose_reward.shape == up_reward.shape == spin_reward.shape
        reward = pose_reward + pose_reward * (up_reward + spin_reward) + effort_reward
        self._tensordict["return"] += reward.unsqueeze(-1)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | (pos[..., 2] < 0.1)
            | (target_dist > 1)
        ) & (self.progress_buf >= self.min_episode_length).unsqueeze(-1)
        return TensorDict(
            {
                "reward": {"drone.reward": reward.unsqueeze(-1)},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )

def lemniscate(t):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    return torch.stack([
        cos_t / (sin_t**2 + 1),
        sin_t * cos_t / (sin_t**2 + 1), 
        torch.zeros_like(t)
    ], dim=-1)


def circle(t):
    return torch.stack([
        (torch.cos(t) - torch.ones_like(t)),
        torch.sin(t),
        torch.zeros_like(t)
    ], dim=-1)

