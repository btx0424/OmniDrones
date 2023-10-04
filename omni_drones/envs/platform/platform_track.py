# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from functorch import vmap
import torch
import torch.distributions as D

import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.objects as objects
from omni.isaac.debug_draw import _debug_draw

import omni_drones.utils.kit as kit_utils

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import cpos, off_diag, others, normalize
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.scene import design_scene
from omni_drones.utils.torch import euler_to_quaternion

from .utils import OveractuatedPlatform, PlatformCfg
from ..utils import lemniscate, scale_time


class PlatformTrack(IsaacEnv):
    r"""
    A cooperative control task where a group of `k` UAVs are connected together by a rigid frame 
    to form an overactuated platform. Each individual UAV, attached by a 2-DoF passive gimbal 
    joint, acts as a thrust generator. The goal for the agents is to 
    collectively control the platform to track a given reference trajectory 
    while orienting the platform towards a given direction.

    ## Observation

    The observation is a `CompositeSpec` containing:

    - `obs_self`` (1, \*): The state of each UAV observed by itself, containing its kinematic
      information with the position being relative to the frame center, and an one-hot
      identity indicating the UAV's index.
    - `obs_others`` (k-1, \*): The observed states of other agents.
    - `obs_frame`:
      - `state_frame`: (1, \*): The state of the frame.
      - `rpos` (3 * `future_traj_steps`): The relative position of the platform to the 
        reference positions in the future `future_traj_steps` time steps.
      - `time_encoding` (optional): The time encoding, which is a 4-dimensional
        vector encoding the current progress of the episode.

    ## Reward

    - `reward_pose`: The reward for the pose error between the platform and 
      the reference (position and orientation).
    - `reward_up`: The reward for the alignment of the platform's up vector and
      the reference up vector.
    - `reward_spin`: Reward computed from the spin of the drone to discourage spinning.
    - `reward_effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    
    The total reward is computed as follows:

     ```{math}
        r = r_\text{pose} + r_\text{pose} * (r_\text{up} + r_\text{spin}) + r_\text{effort}
    ```
      
    ## Config
    
    | Parameter               | Type  | Default       | Description |
    |-------------------------|-------|---------------|-------------|
    | `drone_model`           | str   | "hummingbird" |             |
    | `num_drones`            | int   | 4             |             |
    | `arm_length`            | float | 0.85          |             |
    | `reset_thres`           | float | 0.5           |             |
    | `future_traj_steps`     | int   | 4             |             |
    | `reward_distance_scale` | float | 1.2           |             |
    | `time_encoding`         | bool  | True          |             |
    
    """
    def __init__(self, cfg, headless):
        self.reset_thres = cfg.task.reset_thres
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_potential_weight = cfg.task.reward_potential_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.future_traj_steps = int(cfg.task.future_traj_steps)

        super().__init__(cfg, headless)

        self.platform.initialize()
        self.target_vis = RigidPrimView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.target_vis.initialize()
        
        self.init_vels = torch.zeros_like(self.platform.get_velocities())

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

        sphere = objects.DynamicSphere(
            "/World/envs/env_0/target",
            translation=(0., 0., 2.5),
            radius=0.05,
            color=torch.tensor([1., 0., 0.])
        )
        kit_utils.set_collision_properties(sphere.prim_path, collision_enabled=False)
        kit_utils.set_rigid_body_properties(sphere.prim_path, disable_gravity=True)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape.numel()
        frame_state_dim = 22 + (self.future_traj_steps-1) * 3
        if self.time_encoding:
            self.time_encoding_dim = 4
            frame_state_dim += self.time_encoding_dim
            
        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, drone_state_dim + self.drone.n)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, 13)),
            "obs_frame": UnboundedContinuousTensorSpec((1, frame_state_dim)),
        }).to(self.device)
        observation_central_spec = CompositeSpec({
            "drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim + self.drone.n)),
            "frame": UnboundedContinuousTensorSpec((1, frame_state_dim)),
        }).to(self.device)
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": observation_spec.expand(self.drone.n),
                "observation_central": observation_central_spec,
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": self.drone.action_spec.expand(self.drone.n),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((self.drone.n, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool)
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "observation_central")
        )
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(self.drone.n),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "effort": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

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

        up_target = normalize(torch.randn(len(env_ids), 3, device=self.device)) * 3.
        up_target[..., 2] = up_target[..., 2].abs() + 0.1
        up_target = up_target + self.origin
        self.up_target[env_ids] = up_target
        self.target_vis.set_world_poses(up_target + self.envs_positions[env_ids], env_indices=env_ids)

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
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        self.platform_state = self.platform.get_state()
        drone_pos = self.drone_states[..., :3]
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)

        target_pos = self._compute_traj(self.future_traj_steps, step_size=5)
        target_up = normalize(self.up_target.unsqueeze(1) - self.platform.pos)
        target_platform_rpos = target_pos - self.platform.pos
        target_platform_rup = target_up - self.platform.up
        self.target_distance = torch.norm(target_platform_rpos[..., 0, :], dim=-1, keepdim=True)

        platform_drone_rpos = self.platform.pos - self.drone_states[..., :3]
        platform_state = [
            target_platform_rpos.flatten(-2).unsqueeze(1),
            target_platform_rup,
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
        
        self.up_alignment = torch.sum(self.platform.up * target_up, dim=-1)

        self.stats["pos_error"].lerp_(self.target_distance, (1-self.alpha))
        self.stats["heading_alignment"].lerp_(self.up_alignment, (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference.mean(-1, True), (1-self.alpha))

        return TensorDict({
            "agents": {
                "observation": obs,
                "observation_central": state,
            },
            "stats": self.stats
        }, self.batch_size)

    def _compute_reward_and_done(self):
        platform_vels = self.platform.get_velocities()
        
        reward = torch.zeros(self.num_envs, self.drone.n, device=self.device)
        # reward_pose = 1 / (1 + torch.square(distance * self.reward_distance_scale))
        reward_pose = torch.exp(- self.reward_distance_scale * self.target_distance)
        
        reward_up = torch.square((self.up_alignment + 1) / 2)

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
        )

        done_misbehave = (self.drone_states[..., 2] < 0.2).any(-1, keepdim=True)
        done_hasnan = done_hasnan = torch.isnan(self.drone_states).any(-1)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | done_misbehave
            | done_hasnan.any(-1, keepdim=True)
            | (self.target_distance > self.reset_thres)
        )

        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)

        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(1)},
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

