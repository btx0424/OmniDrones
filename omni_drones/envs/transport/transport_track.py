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


import torch
import torch.distributions as D

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
from omni.isaac.debug_draw import _debug_draw

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import (
    cpos, off_diag, others, euler_to_quaternion, quat_rotate, quat_axis
)
from omni_drones.robots.drone import MultirotorBase

from .utils import TransportationGroup, TransportationCfg
from ..utils import lemniscate, scale_time


class TransportTrack(IsaacEnv):
    r"""
    A cooperative control task where a group of UAVs carry a box-shaped payload connected via
    rigid links. The goal for the agents is to collectively transport the payload to track a 
    reference trajectory.

    ## Observation
    The observation space is specified by a :py:class:`CompositeSpec` containing the following items:

    - `obs_self` (1, \*): The state of each UAV observed by itself, containing its kinematic
      information with the position being relative to the payload. It also includes a one-hot 
      vector indicating each drone's identity.
    - `obs_others` (k-1, \*): The observed states of other agents.
    - `obs_payload` (1, \*): The state of the frame, cotaining its position (relative to the
      reference), rotation (in quaternions and direction vectors), and velocities.
    
    ## Reward
    
    - `pos`: Reward for tracking the trajectory, computed as :math:`\exp(-a * \text{pos_error})`.
    - `seperation`: A factor that penalizes all agents when the minimum seperation is too small.
    - `up`: Reward for keeping the payload upright.
    - `swing`: Reward for avoid large swinging of the payload.
    - `joint_limit`: Reward for kepping the joint states in a reasonalble range to avoid glitchy behaviors.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.

    The total reward is computed as:

    ```{math}
        r = r_\text{seperation} * (r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{swing}) + r_\text{joint\_limit})
    ```

    ## Config

    | Parameter           | Type               | Default       | Description                                                                                                                       |
    |---------------------|--------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------|
    | `drone_model`       | str                | "hummingbird" |                                                                                                                                   |
    | `num_drones`        | int                | 4             |                                                                                                                                   |
    | `safe_distance`     | float              | 0.5           | A threshold value that gives penalty when the minimum seperation between the UAVs is too small.                                   |
    | `mass_scale`        | List[flaot, float] | [0.5, 0.8]    | A tuple of two values that specifies the range of the payload mass to sample from in each episode (as ratio to the drone's mass). |
    | `future_traj_steps` | int                | 4             | The number of future time steps to observe the reference positions along the trajectory.                                          |
    | `reset_thres`       | float              | 0.7           | A threshold value that triggers termination when the payload deviates too far from the reference position.                        |

    """
    def __init__(self, cfg, headless):
        self.reset_thres = cfg.task.reset_thres
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_spin_weight = cfg.task.reward_spin_weight
        self.reward_swing_weight = cfg.task.reward_swing_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.future_traj_steps = int(cfg.task.future_traj_steps)
        self.safe_distance = cfg.task.safe_distance
        super().__init__(cfg, headless)

        self.group.initialize()
        self.payload = self.group.payload_view
        
        self.init_velocities = torch.zeros_like(self.group.get_velocities())
        self.init_joint_pos = self.group.get_joint_positions(clone=True)
        self.init_joint_vel = torch.zeros_like(self.group.get_joint_velocities())

        self.payload_target_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        payload_mass_scale = self.cfg.task.payload_mass_scale
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(payload_mass_scale[0] * self.drone.MASS_0.sum(), device=self.device),
            torch.as_tensor(payload_mass_scale[1] * self.drone.MASS_0.sum(), device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
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

        self.alpha = 0.8

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        group_cfg = TransportationCfg(num_drones=self.cfg.task.num_drones)
        self.group = TransportationGroup(drone=self.drone, cfg=group_cfg)

        scene_utils.design_scene()

        self.group.spawn(translations=[(0, 0, 2.0)], enable_collision=False)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1] + self.drone.n
        payload_state_dim = 19 + (self.future_traj_steps-1) * 3
        if self.time_encoding:
            self.time_encoding_dim = 4
            payload_state_dim += self.time_encoding_dim
        
        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, drone_state_dim)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, 13+1)),
            "obs_payload": UnboundedContinuousTensorSpec((1, payload_state_dim)),
        }).to(self.device)

        state_spec = CompositeSpec(
            drones=UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim)),
            payload=UnboundedContinuousTensorSpec((1, payload_state_dim))
        ).to(self.device)

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": observation_spec.expand(self.drone.n),
                "observation_central": state_spec,
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": torch.stack([self.drone.action_spec] * self.drone.n, dim=0),
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
        )
        
        info_spec = CompositeSpec({
            "payload_mass": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(self.drone.n),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            # "heading_alignment": UnboundedContinuousTensorSpec(1),
            "uprightness": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(self.drone.n),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["info"] = info_spec
        self.observation_spec["stats"] = stats_spec
        self.info = info_spec.zero()
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        traj_w = self.traj_w_dist.sample(env_ids.shape)
        self.traj_w[env_ids] = torch.randn_like(traj_w).sign() * traj_w

        t0 = torch.full([len(env_ids)], self.traj_t0, device=self.device)
        pos = lemniscate(t0, self.traj_c[env_ids]) + self.origin
        rpy = self.init_rpy_dist.sample(env_ids.shape)
        rot = euler_to_quaternion(rpy)

        self.group._reset_idx(env_ids)
        self.group.set_world_poses(pos + self.envs_positions[env_ids], rot, env_ids)
        self.group.set_velocities(self.init_velocities[env_ids], env_ids)

        self.group.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.group.set_joint_velocities(self.init_joint_vel[env_ids], env_ids)

        payload_masses = self.payload_mass_dist.sample(env_ids.shape)
        self.payload.set_masses(payload_masses, env_ids)

        self.info["payload_mass"][env_ids] = payload_masses.unsqueeze(-1).clone()
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

        self.payload_pos, self.payload_rot = self.get_env_poses(self.payload.get_world_poses())
        payload_vels = self.payload.get_velocities()
        self.payload_heading: torch.Tensor = quat_axis(self.payload_rot, axis=0)
        self.payload_up: torch.Tensor = quat_axis(self.payload_rot, axis=2)
        
        self.drone_rpos = torch.vmap(cpos)(self.drone.pos, self.drone.pos)
        self.drone_rpos = torch.vmap(off_diag)(self.drone_rpos)
        self.drone_pdist = torch.norm(self.drone_rpos, dim=-1, keepdim=True)
        payload_drone_rpos = self.payload_pos.unsqueeze(1) - self.drone.pos

        target_pos = self._compute_traj(self.future_traj_steps, step_size=5)
        target_payload_rpos = target_pos - self.payload_pos.unsqueeze(1)
        self.target_distance = torch.norm(target_payload_rpos[..., 0, :], dim=-1, keepdim=True)
        
        self.group.get_state()
        payload_state = [
            target_payload_rpos.flatten(-2),
            self.payload_rot,  # 4
            payload_vels,  # 6
            self.payload_heading,  # 3
            self.payload_up, # 3
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            payload_state.append(t.expand(-1, self.time_encoding_dim))
        payload_state = torch.cat(payload_state, dim=-1).unsqueeze(1)

        obs = TensorDict({}, [self.num_envs, self.drone.n])
        identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)
        obs["obs_self"] = torch.cat(
            [-payload_drone_rpos, self.drone_states[..., 3:], identity], dim=-1
        ).unsqueeze(2) # [..., 1, state_dim]
        obs["obs_others"] = torch.cat(
            [self.drone_rpos, self.drone_pdist, torch.vmap(others)(self.drone_states[..., 3:13])], dim=-1
        ) # [..., n-1, state_dim + 1]
        obs["obs_payload"] = payload_state.expand(-1, self.drone.n, -1).unsqueeze(2) # [..., 1, 22]

        state = TensorDict({}, self.num_envs)
        state["payload"] = payload_state # [..., 1, 22]
        state["drones"] = obs["obs_self"].squeeze(2) # [..., n, state_dim]

        self.stats["pos_error"].lerp_(self.target_distance, (1-self.alpha))
        # self.stats["heading_alignment"].lerp_(heading_alignment, (1-self.alpha))
        self.stats["uprightness"].lerp_(self.payload_up[:, 2].unsqueeze(-1), (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        
        return TensorDict({
            "agents": {
                "observation": obs, 
                "state": state,
            },
            "info": self.info,
            "stats": self.stats
        }, self.num_envs)

    def _compute_reward_and_done(self):
        vels = self.payload.get_velocities()
        joint_positions = (
            self.group.get_joint_positions()[..., :16]
            / self.group.joint_limits[..., :16, 0].abs()
        )
        
        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values

        reward = torch.zeros(self.num_envs, self.drone.n, 1, device=self.device)
        # reward_pos = 1 / (1 + torch.square(distance * self.reward_distance_scale))
        reward_pos = torch.exp(-self.target_distance * self.reward_distance_scale)

        up = self.payload_up[:, 2]
        reward_up = torch.square((up + 1) / 2).unsqueeze(-1)

        spinnage = vels[:, -3:].abs().sum(-1, keepdim=True)
        reward_spin = self.reward_spin_weight * torch.exp(-torch.square(spinnage))

        swing = vels[:, :3].abs().sum(-1, keepdim=True)
        reward_swing = self.reward_swing_weight * torch.exp(-torch.square(swing))

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort).mean(-1, keepdim=True)
        reward_separation = torch.square(separation / self.safe_distance).clamp(0, 1)
        reward_joint_limit = 0.5 * torch.mean(1 - torch.square(joint_positions), dim=-1)

        reward_action_smoothness = self.reward_action_smoothness_weight * -self.drone.throttle_difference
        
        reward[:] = (
            reward_separation * (
                reward_pos 
                + reward_pos * (reward_up + reward_spin + reward_swing) 
                + reward_joint_limit
                + reward_action_smoothness.mean(1, True)
                + reward_effort
            )
        ).unsqueeze(-1)

        done_hasnan = torch.isnan(self.drone_states).any(-1)
        done_fall = self.drone_states[..., 2] < 0.2

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1) 
            | done_fall.any(-1, keepdim=True)
            | done_hasnan.any(-1, keepdim=True)
            | (self.target_distance > self.reset_thres)
        )

        self.stats["return"].add_(reward.mean(1))
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
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
        
        target_pos = torch.vmap(lemniscate)(self.traj_t0 + t, self.traj_c[env_ids])
        target_pos = torch.vmap(quat_rotate)(traj_rot, target_pos) * self.traj_scale[env_ids].unsqueeze(1)

        return self.origin + target_pos
