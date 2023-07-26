from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, normalize
import omni.isaac.core.utils.prims as prim_utils
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from omni.isaac.debug_draw import _debug_draw

from ..utils import lemniscate, scale_time


class TimeEncoding:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, t: torch.Tensor):
        ...


class Fraction(TimeEncoding):
    def __init__(self, max_episode_len, dim: int=4):
        super().__init__(dim)
        self.max_episode_len = max_episode_len
    
    def encode(self, t: torch.Tensor):
        _t = (t / self.max_episode_len).unsqueeze(-1)
        return _t.expand(*t.shape, self.dim)


class Fourier(TimeEncoding):
    def __init__(self, dim: int, seed=0) -> None:
        super().__init__(dim)
    
    def encode(self, t: torch.Tensor):
        ...


class TrackV1(IsaacEnv):
    """
    The UAV need to track a reference trajectory while keeping its heading direction aligned with the flight direction. 
    
    """
    def __init__(self, cfg, headless):
        self.reset_thres = cfg.task.reset_thres
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_motion_smoothness_weight = cfg.task.reward_motion_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.future_traj_len = max(int(cfg.task.future_traj_len), 2)

        super().__init__(cfg, headless)
        
        self.intrinsics = self.cfg.task.intrinsics
        self.wind = self.cfg.task.wind

        self.drone.initialize()
        randomization = self.cfg.task.get("randomization", None)
        if randomization is not None:
            if "drone" in self.cfg.task.randomization:
                self.drone.setup_randomization(self.cfg.task.randomization["drone"])
        
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        self.traj_c_dist = D.Uniform(
            torch.tensor(-0.6, device=self.device),
            torch.tensor(0.6, device=self.device)
        )
        self.traj_scale_dist = D.Uniform(
            torch.tensor([1.8, 1.8, 1.], device=self.device),
            torch.tensor([3.2, 3.2, 1.5], device=self.device)
        )
        self.traj_w_dist = D.Uniform(
            torch.tensor(0.8, device=self.device),
            torch.tensor(1.1, device=self.device)
        )
        self.origin = torch.tensor([0., 0., 2.], device=self.device)

        self.traj_t0 = torch.pi / 2
        self.traj_c = torch.zeros(self.num_envs, device=self.device)
        self.traj_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.traj_w = torch.ones(self.num_envs, device=self.device)

        self.ref_pos = torch.zeros(self.num_envs, self.future_traj_len, 3, device=self.device)
        self.ref_heading = torch.zeros(self.num_envs, 2, device=self.device)

        self.alpha = 0.8
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

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        obs_dim = drone_state_dim + 3 * (self.future_traj_len-1) + 2
        if self.time_encoding:
            self.time_encoding = Fraction(self.max_episode_length)
            obs_dim += self.time_encoding.dim
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((1, obs_dim))
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((1, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "tracking_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        # info_spec = self.drone.info_spec.to(self.device)
        # self.observation_spec["info"] = info_spec
        self.observation_spec["stats"] = stats_spec
        # self.info = info_spec.zero()
        self.stats = stats_spec.zero()
        

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        traj_w = self.traj_w_dist.sample(env_ids.shape)
        self.traj_w[env_ids] = torch.randn_like(traj_w).sign() * traj_w

        t0 = torch.full((len(env_ids),), self.traj_t0, device=self.device)
        pos_0 = lemniscate(t0, self.traj_c[env_ids]) + self.origin
        pos_1 = lemniscate(t0+self.dt*traj_w, self.traj_c[env_ids]) + self.origin
        traj_heading = normalize(pos_1 - pos_0)
        rpy = torch.stack([
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device),
            torch.arctan2(traj_heading[:, 1], traj_heading[:, 0])
        ], dim=-1)
        rot = euler_to_quaternion(rpy)
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        self.drone.set_world_poses(
            pos_0 + self.envs_positions[env_ids], rot, env_ids
        )
        self.drone.set_velocities(vel, env_ids)

        self.stats[env_ids] = 0.

        # self.info[env_ids] = self.drone.info[env_ids]

        if self._should_render(0) and (env_ids == self.central_env_idx).any() :
            # visualize the trajectory
            self.draw.clear_lines()

            traj_vis = self._compute_traj(self.max_episode_length, self.central_env_idx.unsqueeze(0))[0]
            traj_vis = traj_vis + self.envs_positions[self.central_env_idx]
            point_list_0 = traj_vis[:-1].tolist()
            point_list_1 = traj_vis[1:].tolist()
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [1 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()

        self.ref_pos[:] = self._compute_traj(self.future_traj_len, step_size=5)
        self.ref_heading[:] = normalize(self.ref_pos[:, 1, :2] - self.ref_pos[:, 0, :2])

        self.rpos = self.ref_pos - self.root_state[..., :3]
        obs = [
            self.rpos.flatten(1).unsqueeze(1),
            self.ref_heading.unsqueeze(1) - normalize(self.drone.heading[..., :2]),
            self.root_state[..., 3:],
        ]
        if self.time_encoding:
            obs.append(self.time_encoding.encode(self.progress_buf).unsqueeze(1))
        
        if self.intrinsics:
            obs.append(self.drone.get_info())

        obs = torch.cat(obs, dim=-1)

        return TensorDict({
            "agents": {
                "observation": obs,
            },
            "stats": self.stats
        }, self.batch_size)

    def _compute_reward_and_done(self):
        pos_error = torch.norm(self.rpos[:, [0]], dim=-1)
        heading_alignment = (
            self.ref_heading.unsqueeze(1) * normalize(self.drone.heading[..., :2])
        ).sum(-1)
        
        reward_pose = torch.exp(-self.reward_distance_scale * pos_error)
        reward_heading = heading_alignment
    
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        reward = (
            reward_pose 
            + reward_pose * (reward_heading) 
            + reward_effort
            + reward_action_smoothness
        )
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        self.stats["tracking_error"].add_(pos_error)
        self.stats["heading_alignment"].add_(heading_alignment)
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | (self.drone.pos[..., 2] < 0.1)
            | (pos_error > self.reset_thres)
        ) 
        
        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": done,
            },
            self.batch_size,
        )
    
    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1.):
        if env_ids is None:
            env_ids = ...
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        t = self.traj_t0 + scale_time(self.traj_w[env_ids].unsqueeze(1) * t * self.dt)
        traj_rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, t.shape[1], 4)
        
        ref_pos = vmap(lemniscate)(t, self.traj_c[env_ids])
        ref_pos = vmap(torch_utils.quat_rotate)(traj_rot, ref_pos) * self.traj_scale[env_ids].unsqueeze(1)

        return self.origin + ref_pos

