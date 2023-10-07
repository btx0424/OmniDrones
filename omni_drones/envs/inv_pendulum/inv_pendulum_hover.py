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


import functorch
import torch
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec

import omni.isaac.core.objects as objects

import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, normalize
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView

from .utils import create_pendulum


class InvPendulumHover(IsaacEnv):
    r"""
    An intermidiate control task where a classic inverted pendulum is based on the UAV.
    We refer the the ball at the end of pendulum as *payload*. The goal for the agent 
    is to keep balance while maintaining its position around a target position.

    ## Observation
    - `drone_payload_rpos` (3): The position of the drone relative to the payload's position.
    - `root_state` (19 + num_rotors): The basic information of the drone, 
      containing its rotation (in quaternion), velocities (linear and angular), 
      heading and up vectors, and the current throttle.
    - `target_payload_rpos` (3): The position of the reference relative to the payload's position.
    - `payload_vel` (6): The linear and angular velocities of the payload.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional
      vector encoding the current progress of the episode.
 
    ## Reward

    - `pos`: Reward for maintaining the position of the payload around the target position.
    - `bar_up`: Reward for keeping the bar up.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.
    - `swing`: Reward computed from the swing of the payload to discourage swinging.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{bar_up} + r_\text{pos} + r_\text{bar_up} * (r_\text{spin} + r_\text{swing}) + r_\text{effort}
    ```

    ## Episode End
    The episode ends when the bar falls beyond a certain angle, or when the
    drone gets too close to the ground, or when the distance between the payload 
    and the target exceeds a threshold, or when the maximum episode length
    is reached.

    ## Config

    | Parameter               | Type  | Default       | Description |
    |-------------------------|-------|---------------|-------------|
    | `drone_model`           | str   | "hummingbird" | Specifies the model of the drone being used in the environment. |
    | `bar_length`            | float | 1.0           | Length of the pendulum's bar. |
    | `reward_distance_scale` | float | 1.2           | Scales the reward based on `target_payload_rpos`. |
    | `time_encoding`         | bool  | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |

    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding

        self.bar_length = 1.
        super().__init__(cfg, headless)

        self.drone.initialize()

        # create and initialize additional views
        self.payload = RigidPrimView(
            f"/World/envs/env_*/{self.drone.name}_*/payload",
        )
        self.payload.initialize()
        self.bar = RigidPrimView(
            f"/World/envs/env_*/{self.drone.name}_*/bar",
        )
        self.bar.initialize()

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())

        self.init_pos_dist = D.Uniform(
            torch.tensor([-1.5, -1.5, 1.0], device=self.device),
            torch.tensor([1.5, 1.5, 2.0], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.15, -.15, 0.], device=self.device) * torch.pi,
            torch.tensor([0.15, 0.15, 2], device=self.device) * torch.pi
        )
        payload_mass_scale = self.cfg.task.payload_mass_scale
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(payload_mass_scale[0] * self.drone.MASS_0, device=self.device),
            torch.as_tensor(payload_mass_scale[1] * self.drone.MASS_0, device=self.device)
        )
        self.bar_mass_dist = D.Uniform(
            torch.as_tensor(self.cfg.task.bar_mass_min, device=self.device),
            torch.as_tensor(self.cfg.task.bar_mass_max, device=self.device)
        )

        self.payload_target_pos = torch.tensor([0., 0., 2.5], device=self.device)
        
        self.alpha = 0.8

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

        self.drone.spawn(translations=[(0.0, 0.0, 2.)])
        create_pendulum(f"/World/envs/env_0/{self.drone.name}_0", self.bar_length, 0.04)

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
        observation_dim = self.drone.state_spec.shape[-1] + 12
        if self.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim
        
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim)) ,
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool)
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
            "pos_error": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        
        drone_pos = self.init_pos_dist.sample((*env_ids.shape, 1))
        drone_rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.drone.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.drone.set_joint_velocities(self.init_joint_vels[env_ids], env_ids)

        payload_mass = self.payload_mass_dist.sample(env_ids.shape)
        self.payload.set_masses(payload_mass, env_ids)
        bar_mass = self.bar_mass_dist.sample(env_ids.shape)
        self.bar.set_masses(bar_mass, env_ids)

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.info["drone_state"][:] = self.drone_state[..., :13]
        self.drone_up = self.drone_state[..., 16:19]
        payload_pos, payload_rot = self.get_env_poses(self.payload.get_world_poses())
        self.payload_vels = self.payload.get_velocities()

        # relative position and heading
        self.drone_payload_rpos = self.drone_state[..., :3] - payload_pos.unsqueeze(1)
        target_payload_rpos = self.payload_target_pos - payload_pos.unsqueeze(1)

        obs = [
            self.drone_payload_rpos, # 3
            self.drone_state,
            target_payload_rpos, # 3
            self.payload_vels.unsqueeze(1), # 6
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        self.pos_error = torch.norm(target_payload_rpos, dim=-1)

        self.stats["pos_error"].lerp_(self.pos_error, (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))

        return TensorDict({
            "agents":{
                "observation": obs,
            },
            "stats": self.stats,
            "info": self.info,
        }, self.batch_size)

    def _compute_reward_and_done(self):        
        # reward_pos = 1.0 / (1.0 + torch.square(self.reward_distance_scale * self.pos_error))
        reward_pos = torch.exp(-self.reward_distance_scale * self.pos_error)

        reward_bar_up = normalize(-self.drone_payload_rpos)[..., 2]

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)

        spin = torch.square(self.drone.vel[..., -1])
        reward_spin = 1. / (1.0 + torch.square(spin))

        swing = torch.norm(self.payload_vels[..., :3], dim=-1, keepdim=True)
        reward_swing = torch.exp(-swing)

        reward = (
            reward_bar_up + reward_pos
            + reward_bar_up * (reward_spin + reward_swing) 
            + reward_effort
        )
        
        done_misbehave = (self.drone.pos[..., 2] < 0.2) | (reward_bar_up < 0.2)
        done_hasnan = torch.isnan(self.drone_state).any(-1)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | done_misbehave
            | done_hasnan
            | (self.pos_error > 3.3)
        )
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": done,
            },
            self.batch_size,
        )
