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
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec

import omni_drones.utils.kit as kit_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import cpos, off_diag, others
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.scene import design_scene
from omni_drones.utils.torch import euler_to_quaternion

from .utils import OveractuatedPlatform, PlatformCfg



class PlatformHover(IsaacEnv):
    r"""
    A cooperative control task where a group of `k` UAVs are connected together by a
    rigid frame to form an overactuated platform. Each individual UAV, attached
    by a 2-DoF passive gimbal joint, acts as a thrust generator.
    The goal for the agents is to make the platform hover at a reference pose 
    (position and attitude).

    ## Observation
    The observation is a `CompositeSpec` containing the following items:

    - `obs_self` (1, \*): The state of each UAV observed by itself, containing its kinematic
      information with the position being relative to the frame center, and an one-hot
      identity indicating the UAV's index.
    - `obs_others` (k-1, \*): The observed states of other agents.
    - `obs_frame`:
      - `state_frame`: (1, \*): The state of the frame.
      - `rpos` (3): The relative position of the platform to the reference positions.
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
    | `reward_distance_scale` | float | 1.2           |             |
    | `time_encoding`         | bool  | True          |             |

    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        
        self.num_drones = cfg.task.num_drones
        self.arm_length = cfg.task.arm_length
        self.joint_damping  = cfg.task.joint_damping
        super().__init__(cfg, headless)

        self.platform.initialize()

        self.target_vis = RigidPrimView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        ).initialize()
        
        self.init_vels = torch.zeros_like(self.platform.get_velocities())
        self.init_joint_pos = self.platform.get_joint_positions(clone=True)
        self.init_joint_vel = torch.zeros_like(self.platform.get_joint_velocities())

        self.init_pos_dist = D.Uniform(
            torch.tensor([-3., -3., 1.5], device=self.device),
            torch.tensor([3.0, 3.0, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2], device=self.device) * torch.pi
        )
        self.target_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.0], device=self.device),
            torch.tensor([0.2, 0.2, 2.0], device=self.device)
        )
        self.target_pos = torch.tensor([0., 0., 2.25], device=self.device)
        self.target_heading =  torch.zeros(self.num_envs, 3, device=self.device)
        self.target_up = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_distance = torch.zeros(self.num_envs, 1, device=self.device)

        self.alpha = 0.8

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        platform_cfg = PlatformCfg(
            num_drones=self.num_drones,
            arm_length=self.arm_length,
            joint_damping=self.joint_damping
        )
        self.platform = OveractuatedPlatform(
            cfg=platform_cfg,
            drone=self.drone,
        )
        self.platform.spawn(
            translations=[0., 0., 2.],
            enable_collision=True
        )

        # for visulization
        target_prim_path = self.platform._create_frame(
            "/World/envs/env_0/target",
            enable_collision=False
        ).GetPath().pathString
        kit_utils.set_rigid_body_properties(target_prim_path, disable_gravity=True)

        design_scene()
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape.numel()
        frame_state_dim = 25
        if self.time_encoding:
            self.time_encoding_dim = 4
            frame_state_dim += self.time_encoding_dim
            
        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, drone_state_dim + self.drone.n)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, 13)),
            "state_frame": UnboundedContinuousTensorSpec((1, frame_state_dim)),
        }).to(self.device)
        observation_central_spec = CompositeSpec({
            "state_drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim + self.drone.n)),
            "state_frame": UnboundedContinuousTensorSpec((1, frame_state_dim)),
        }).to(self.device)
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": observation_spec.expand(self.drone.n),
                "observation_central": observation_central_spec,
            },
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
        self.drone._reset_idx(env_ids)

        platform_pos = self.init_pos_dist.sample(env_ids.shape) 
        platform_rpy = self.init_rpy_dist.sample(env_ids.shape)
        platform_rot = euler_to_quaternion(platform_rpy)
        platform_heading = torch_utils.quat_axis(platform_rot, 0)
        platform_up = torch_utils.quat_axis(platform_rot,  2)
        self.platform.set_world_poses(
            platform_pos + self.envs_positions[env_ids], 
            platform_rot, env_indices=env_ids
        )
        self.platform.set_velocities(self.init_vels[env_ids], env_ids)

        self.platform.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.platform.set_joint_velocities(self.init_joint_vel[env_ids], env_ids)

        target_rpy = self.target_rpy_dist.sample(env_ids.shape)
        target_rot = euler_to_quaternion(target_rpy)
        target_heading = torch_utils.quat_axis(target_rot, 0)
        target_up = torch_utils.quat_axis(target_rot, 2)
        self.target_heading[env_ids] = target_heading
        self.target_up[env_ids] = target_up
        
        self.target_vis.set_world_poses(
            self.target_pos + self.envs_positions[env_ids],
            orientations=target_rot,
            env_indices=env_ids
        )

        self.stats[env_ids] = 0.
        distance = torch.cat([
            self.target_pos - platform_pos,
            target_heading - platform_heading,
            target_up - platform_up
        ], dim=-1).norm(dim=-1, keepdim=True)
        self.last_distance[env_ids] = distance

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        drone_pos = self.drone_states[..., :3]
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)

        self.platform_state = self.platform.get_state()

        self.target_platform_rpos = self.target_pos - self.platform.pos
        self.target_platform_rheading = self.target_heading.unsqueeze(1) - self.platform.heading
        self.target_platform_rup = self.target_up.unsqueeze(1) - self.platform.up
        self.target_platform_rpose = torch.cat([
            self.target_platform_rpos,
            self.target_platform_rheading,
            self.target_platform_rup
        ], dim=-1)

        platform_drone_rpos = self.platform.pos - self.drone_states[..., :3]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            platform_state = torch.cat([
                self.target_platform_rpose, # 9
                self.platform_state[..., 3:],
                t.expand(-1, self.time_encoding_dim).unsqueeze(1)
            ], dim=-1) # [num_envs, 1, 25+time_encoding_dim]
        else:
            platform_state = torch.cat([
                self.target_platform_rpose, # 9
                self.platform_state[..., 3:]
            ], dim=-1) # [num_envs, 1, 25]

        identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)

        obs = TensorDict({}, [self.num_envs, self.drone.n])
        obs["obs_self"] = torch.cat(
            [-platform_drone_rpos, self.drone_states[..., 3:], identity], dim=-1
        ).unsqueeze(2)
        obs["obs_others"] = torch.cat(
            [self.drone_rpos, vmap(others)(self.drone_states[..., 3:13])], dim=-1
        )
        obs["state_frame"] = platform_state.unsqueeze(1).expand(-1, self.drone.n, 1, -1)

        state = TensorDict({}, [self.num_envs])
        state["state_drones"] = obs["obs_self"].squeeze(2)    # [num_envs, drone.n, drone_state_dim]
        state["state_frame"] = platform_state                # [num_envs, 1, platform_state_dim]
        
        self.pos_error = torch.norm(self.target_platform_rpos, dim=-1)
        self.heading_alignment = torch.sum(self.platform.heading * self.target_heading.unsqueeze(1), dim=-1)
        
        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "observation_central": state,
                },
                "stats": self.stats
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        platform_vels = self.platform.get_velocities()

        distance = torch.norm(self.target_platform_rpose, dim=-1)
        
        reward = torch.zeros(self.num_envs, self.drone.n, device=self.device)
        # reward_pose = 1 / (1 + torch.square(distance * self.reward_distance_scale))
        reward_pose = torch.exp(- self.reward_distance_scale * distance)
        
        up = torch.sum(self.platform.up * self.target_up.unsqueeze(1), dim=-1)
        reward_up = torch.square((up + 1) / 2)

        spinnage = platform_vels[:, -3:].abs().sum(-1)
        reward_spin = 1. / (1 + torch.square(spinnage))
        
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort).mean(-1, keepdim=True)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference).mean(-1, keepdim=True)

        self.last_distance[:] = distance
        assert reward_pose.shape == reward_up.shape == reward_action_smoothness.shape

        reward[:] = (
            reward_pose
            + reward_pose * (reward_up + reward_spin) 
            + reward_effort
            + reward_action_smoothness
        )

        done_misbehave = (self.drone_states[..., 2] < 0.2).any(-1, keepdim=True) | (distance > 5.0)
        done_hasnan = done_hasnan = torch.isnan(self.drone_states).any(-1)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | done_misbehave
            | done_hasnan.any(-1, keepdim=True)
        )

        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)
        self.stats["pos_error"].lerp_(self.pos_error, (1-self.alpha))
        self.stats["heading_alignment"].lerp_(self.heading_alignment, (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference.mean(-1, True), (1-self.alpha))
        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(1)
                },
                "done": done,
            },
            self.batch_size,
        )

