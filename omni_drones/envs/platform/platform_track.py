from functorch import vmap
import torch
import torch.distributions as D

import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.prims as prim_utils
import omni_drones.utils.kit as kit_utils
from omni.isaac.debug_draw import _debug_draw

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import cpos, off_diag, others
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.scene import design_scene
from omni_drones.utils.torch import euler_to_quaternion

from .utils import OveractuatedPlatform, PlatformCfg
from ..utils import lemniscate, scale_time


class PlatformTrack(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reset_thres = self.cfg.task.reset_thres
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = self.cfg.task.reward_action_smoothness_weight
        self.reward_motion_smoothness_weight = self.cfg.task.reward_motion_smoothness_weight
        self.reward_potential_weight = self.cfg.task.reward_potential_weight
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.time_encoding = self.cfg.task.time_encoding
        self.future_traj_len = int(self.cfg.task.future_traj_len)

        self.platform.initialize()
        
        self.init_vels = torch.zeros_like(self.platform.get_velocities())

        drone_state_dim = self.drone.state_spec.shape.numel()
        frame_state_dim = 19 + (self.future_traj_len-1) * 3
        if self.time_encoding:
            self.time_encoding_dim = 4
            frame_state_dim += self.time_encoding_dim
            
        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, drone_state_dim + self.drone.n)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, 13)),
            "obs_frame": UnboundedContinuousTensorSpec((1, frame_state_dim)),
        }).to(self.device)
        state_spec = CompositeSpec({
            "drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim + self.drone.n)),
            "frame": UnboundedContinuousTensorSpec((1, frame_state_dim)),
        }).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.drone.n,
            observation_spec,
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
            state_spec
        )

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2], device=self.device) * torch.pi
        )
        self.up_target_dist = D.Uniform(
            torch.tensor([-3., -3., 1.5], device=self.device),
            torch.tensor([3.0, 3.0, 2.5], device=self.device)
        )
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        self.traj_c_dist = D.Uniform(
            torch.tensor(-0.6, device=self.device),
            torch.tensor(0.6, device=self.device)
        )
        self.traj_scale_dist = D.Uniform(
            torch.tensor([2.4, 2.4, 1.2], device=self.device),
            torch.tensor([3.6, 3.6, 1.6], device=self.device)
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

        self.up_target = torch.zeros(self.num_envs, 3, device=self.device)

        self.alpha = 0.8
        stats_spec = CompositeSpec({
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "effort": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
            "motion_smoothness": UnboundedContinuousTensorSpec(1)
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        platform_cfg = PlatformCfg(
            num_drones=self.cfg.task.num_drones,
            arm_length=self.cfg.task.arm_length,
            joint_damping=self.cfg.task.joint_damping
        )
        self.platform = OveractuatedPlatform(
            cfg=platform_cfg,
            drone=self.drone,
        )
        self.platform.spawn(
            translations=[0., 0., 2.],
            enable_collision=True
        )

        design_scene()
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        traj_w = self.traj_w_dist.sample(env_ids.shape)
        self.traj_w[env_ids] = torch.randn_like(traj_w).sign() * traj_w

        t0 = torch.full([len(env_ids)], self.traj_t0, device=self.device)
        platform_pos = lemniscate(t0, self.traj_c[env_ids]) + self.origin
        platform_rpy = self.init_rpy_dist.sample(env_ids.shape)
        platform_rot = euler_to_quaternion(platform_rpy)

        self.platform._reset_idx(env_ids)
        self.platform.set_world_poses(
            platform_pos + self.envs_positions[env_ids], 
            platform_rot, env_indices=env_ids
        )
        self.platform.set_velocities(self.init_vels[env_ids], env_ids)
        self.up_target[env_ids] = self.up_target_dist.sample(env_ids)

        self.stats[env_ids] = 0.

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
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        self.platform_state = self.platform.get_state()
        drone_pos = self.drone_states[..., :3]
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)

        target_pos = self._compute_traj(self.future_traj_len, step_size=5)
        target_platform_rpos = target_pos - self.platform.pos
        self.target_distance = torch.norm(target_platform_rpos[..., 0, :], dim=-1, keepdim=True)

        platform_drone_rpos = self.platform.pos - self.drone_states[..., :3]
        platform_state = [
            target_platform_rpos.flatten(-2).unsqueeze(1),
            self.platform_state[..., 3:],
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            platform_state.append(t.expand(-1, 1, self.time_encoding_dim))
        platform_state = torch.cat(platform_state, dim=-1)

        identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)

        obs = TensorDict({}, [self.num_envs, self.drone.n])
        obs["obs_self"] = torch.cat(
            [-platform_drone_rpos, self.drone_states[..., 3:], identity], dim=-1
        ).unsqueeze(2)
        obs["obs_others"] = torch.cat(
            [self.drone_rpos, vmap(others)(self.drone_states[..., 3:13])], dim=-1
        )
        obs["obs_frame"] = platform_state.unsqueeze(1).expand(-1, self.drone.n, 1, -1)

        state = TensorDict({}, [self.num_envs])
        state["drones"] = obs["obs_self"].squeeze(2)    # [num_envs, drone.n, drone_state_dim]
        state["frame"] = platform_state                # [num_envs, 1, platform_state_dim]
        
        self.stats["pos_error"].lerp_(self.target_distance, (1-self.alpha))
        # self.stats["heading_alignment"].lerp_(heading_alignment, (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference.mean(-1, True), (1-self.alpha))
        self.motion_smoothness = (
            self.platform.get_linear_smoothness() 
            + self.platform.get_angular_smoothness()
        )
        self.stats["motion_smoothness"].lerp_(self.motion_smoothness, (1-self.alpha))

        return TensorDict({
            "drone.obs": obs,
            "drone.state": state,
            "stats": self.stats
        }, self.batch_size)

    def _compute_reward_and_done(self):
        platform_vels = self.platform.get_velocities()
        
        reward = torch.zeros(self.num_envs, self.drone.n, 1, device=self.device)
        # reward_pose = 1 / (1 + torch.square(distance * self.reward_distance_scale))
        reward_pose = torch.exp(- self.reward_distance_scale * self.target_distance)
        
        up = torch.sum(self.platform.up * self.target_up.unsqueeze(1), dim=-1)
        reward_up = torch.square((up + 1) / 2)

        spinnage = platform_vels[:, -3:].abs().sum(-1)
        reward_spin = 1. / (1 + torch.square(spinnage))
        
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort).mean(-1, keepdim=True)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference).mean(-1, keepdim=True)

        assert reward_pose.shape == reward_up.shape == reward_action_smoothness.shape

        reward[:] = (
            reward_pose
            + reward_pose * (reward_up + reward_spin) 
            + reward_effort
            + reward_action_smoothness
        ).unsqueeze(1)

        done_misbehave = (self.drone_states[..., 2] < 0.2).any(-1, keepdim=True)
        done_hasnan = done_hasnan = torch.isnan(self.drone_states).any(-1)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | done_misbehave
            | done_hasnan.any(-1, keepdim=True)
            | (self.target_distance > self.reset_thres)
        )

        self._tensordict["return"] += reward
        return TensorDict(
            {
                "reward": {"drone.reward": reward},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )

    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1.):
        if env_ids is None:
            env_ids = ...
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        t = scale_time(self.traj_w[env_ids].unsqueeze(1) * t * self.dt)
        traj_rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, t.shape[1], 4)
        
        target_pos = vmap(lemniscate)(self.traj_t0 + t, self.traj_c[env_ids])
        target_pos = vmap(torch_utils.quat_rotate)(traj_rot, target_pos) * self.traj_scale[env_ids].unsqueeze(1)

        return self.origin + target_pos

