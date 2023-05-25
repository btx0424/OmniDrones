from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion
import omni.isaac.core.utils.prims as prim_utils
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec
from omni.isaac.debug_draw import _debug_draw

class Tracking(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reset_thres = self.cfg.task.reset_thres
        self.reward_action_smoothness_weight = self.cfg.task.reward_action_smoothness_weight
        self.reward_motion_smoothness_weight = self.cfg.task.reward_motion_smoothness_weight
        self.time_encoding = self.cfg.task.time_encoding
        self.future_traj_len = int(self.cfg.task.future_traj_len)
        assert self.future_traj_len > 0

        self.drone.initialize()
        if "randomization" in self.cfg.task:
            if "drone" in self.cfg.task.randomization:
                self.drone.setup_randomization(self.cfg.task.randomization["drone"])
        
        drone_state_dim = self.drone.state_spec.shape[-1]
        obs_dim = drone_state_dim + 3 * (self.future_traj_len-1)
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_dim += self.time_encoding_dim
        
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            UnboundedContinuousTensorSpec(obs_dim).to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )
        
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        self.traj_c_dist = D.Uniform(
            torch.tensor(0, device=self.device),
            torch.tensor(0.6, device=self.device)
        )
        self.traj_scale_dist = D.Uniform(
            torch.tensor([1., 1., 1.], device=self.device),
            torch.tensor([2., 2., 1.5], device=self.device)
        )
        self.w_dist = D.Uniform(
            torch.tensor(0.8, device=self.device),
            torch.tensor(1.2, device=self.device)
        )
        self.origin = torch.tensor([0., 0., 2.], device=self.device)
        self.phase = torch.pi / 2

        self.traj_c = torch.zeros(self.num_envs, device=self.device)
        self.traj_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_rot = torch.zeros(self.num_envs, 4, device=self.device)

        self.target_pos = torch.zeros(self.num_envs, self.future_traj_len, 3, device=self.device)
        # self.target_vel = torch.zeros(self.num_envs, self.future_traj_len, 3, device=self.device)
        self.w = torch.ones(self.num_envs, device=self.device)

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
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
        self.drone._reset_idx(env_ids)
        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)

        t0 = torch.zeros(len(env_ids), device=self.device)
        pos = lemniscate(t0 + self.phase, self.traj_c[env_ids])
        rot = self.init_rpy_dist.sample(env_ids.shape)
        vel = torch.zeros(self.num_envs, 1, 6, device=self.device)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(vel, env_ids)


    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()

        ts = self.progress_buf.unsqueeze(1) + torch.arange(self.future_traj_len, device=self.device)
        ts = self.phase + self.w * t
        
        target_pos = vmap(lemniscate)(ts, self.traj_c)
        target_pos = vmap(torch_utils.quat_rotate)(target_pos) * self.traj_scale
        self.target_pos[:] = self.origin + target_pos
        
        self.rpos = self.target_pos - self.root_state[..., :3]
        obs = [
            self.rpos.flatten(1).unsqueeze(1),
            self.root_state[..., 3:],
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

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

def lemniscate(t, c):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    sin2p1 = torch.square(sin_t) + 1

    x = torch.stack([
        cos_t, sin_t * cos_t, c * sin_t
    ], dim=-1) / sin2p1.unsqueeze(-1)

    return x

