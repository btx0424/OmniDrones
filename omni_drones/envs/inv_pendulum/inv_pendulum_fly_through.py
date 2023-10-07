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
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    UnboundedContinuousTensorSpec, 
    CompositeSpec, 
    DiscreteTensorSpec
)

import omni.isaac.core.objects as objects
from omni.isaac.debug_draw import _debug_draw

import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, normalize
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView

from .utils import create_pendulum
from ..utils import create_obstacle

class InvPendulumFlyThrough(IsaacEnv):
    r"""
    An intermidiate control task where a classic inverted pendulum is based on the UAV.
    We refer the the ball at the end of pendulum as *payload*. The goal for the agent 
    is to pass between two horizontal bars and keep balance while maintaining its position 
    around a target position. If the distance between the two horizontal bars is less than 
    the length of the rod, the agent need learn to swing the inverted pendulum to pass through.

    ## Observation
    - `drone_payload_rpos` (3): The position of the drone relative to the payload's position.
    - `root_state` (19 + num_rotors): The basic information of the drone, 
      containing its rotation (in quaternion), velocities (linear and angular), 
      heading and up vectors, and the current throttle.
    - `target_payload_rpos` (3): The position of the reference relative to the payload's position.
    - `payload_vel` (6): The linear and angular velocities of the payload.
    - `obstacle_drone_rpos` (2 * 2 = 4): The position of the two bars relative to the drone's position.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional
      vector encoding the current progress of the episode.

    ## Reward
    - `pos`: Reward for maintaining the final position of the payload around the target position. 
    - `bar_up`: Reward for keeping the bar up.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.
    - `swing`: Reward computed from the swing of the payload to discourage swinging.
    - `collision`: Reward for avoiding collisions with horizontal bars.

    The total reward is computed as follows: 

    ```{math}  
        r = [r_\text{pos} + r_\text{pos} * (r_\text{bar_up} + r_\text{spin} + r_\text{swing}) + r_\text{effort}] * (1 - r_\text{collision})
    ```
        
    ## Episode End

    The episode ends when the bar falls beyond a certain angle, or when the
    drone gets too close to the ground, or when the drone goes too far away horizontally, 
    or when the payload goes too far away vertically, or when the maximum episode length 
    is reached, or (optional) when the drone collides with any obstacle.
    

    ## Config

    | Parameter               | Type                | Default       | Description |
    |-------------------------|---------------------|---------------|-------------|
    | `drone_model`           | str                 | "hummingbird" | Specifies the model of the drone being used in the environment. |
    | `reset_on_collision`    | bool                | False         | Indicates if the episode should reset when the drone collides with an obstacle. |
    | `bar_length`            | float               | 0.85          | Length of the pendulum's bar. |
    | `reward_distance_scale` | float               | 1.2           | Scales the reward based on the distance between the payload and its target. |
    | `time_encoding`         | bool                | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
    | `obstacle_spacing`      | tuple[float, float] | [0.9, 1.2]    | Specifies the minimum and maximum distance between two horizontal bars (obstacles) in the environment. |

    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reset_on_collision = cfg.task.reset_on_collision
        self.time_encoding = cfg.task.time_encoding
        self.obstacle_spacing = cfg.task.obstacle_spacing

        super().__init__(cfg, headless)

        self.drone.initialize()

        self.obstacles = RigidPrimView(
            "/World/envs/env_*/obstacle_*",
            reset_xform_properties=False,
            shape=[self.num_envs, -1],
            track_contact_forces=True
        )
        self.obstacles.initialize()
        self.payload = RigidPrimView(
            f"/World/envs/env_*/{self.drone.name}_*/payload",
        )
        self.payload.initialize()
        self.bar = RigidPrimView(
            f"/World/envs/env_*/{self.drone.name}_*/bar",
        )
        self.bar.initialize()

        self.payload_target_vis = RigidPrimView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.payload_target_vis.initialize()

        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())
        self.obstacle_pos = self.get_env_poses(self.obstacles.get_world_poses())[0]

        self.init_pos_dist = D.Uniform(
            torch.tensor([-2.2, 0, 1.0], device=self.device),
            torch.tensor([-1.0, 0, 2.0], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.15, -.15, 0.], device=self.device) * torch.pi,
            torch.tensor([0.15, 0.15, 2], device=self.device) * torch.pi
        )
        self.obstacle_spacing_dist = D.Uniform(
            torch.tensor(self.obstacle_spacing[0], device=self.device),
            torch.tensor(self.obstacle_spacing[1], device=self.device)
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

        self.alpha = 0.8

        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.payload_traj_vis = []
        self.drone_traj_vis = []

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

        create_obstacle(
            "/World/envs/env_0/obstacle_0", 
            prim_type="Capsule",
            translation=(0., 0., 1.2),
            attributes={"axis": "Y", "radius": 0.04, "height": 5}
        )
        create_obstacle(
            "/World/envs/env_0/obstacle_1", 
            prim_type="Capsule",
            translation=(0., 0., 2.2),
            attributes={"axis": "Y", "radius": 0.04, "height": 5}
        )

        self.drone.spawn(translations=[(0.0, 0.0, 2.)])
        create_pendulum(
            f"/World/envs/env_0/{self.drone.name}_0", 
            self.cfg.task.bar_length,
            payload_radius=0.04
        )

        self.payload_target_pos = torch.tensor([1.5, 0., 2.3], device=self.device)
        sphere = objects.DynamicSphere(
            "/World/envs/env_0/target",
            translation=self.payload_target_pos,
            radius=0.05,
            color=torch.tensor([1., 0., 0.])
        )
        kit_utils.set_collision_properties(sphere.prim_path, collision_enabled=False)
        kit_utils.set_rigid_body_properties(sphere.prim_path, disable_gravity=True)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        observation_dim = self.drone.state_spec.shape[-1] + 12 + 4
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
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
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
            "collision": UnboundedContinuousTensorSpec(1),
            "success": DiscreteTensorSpec(2, (1,), dtype=bool),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

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

        obstacle_spacing = self.obstacle_spacing_dist.sample(env_ids.shape)
        obstacle_pos = torch.zeros(len(env_ids), 2, 3, device=self.device)
        obstacle_pos[:, :, 2] = 1.2
        obstacle_pos[:, 1, 2] += obstacle_spacing
        self.obstacles.set_world_poses(
            obstacle_pos + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids
        )
        self.obstacle_pos[env_ids] = obstacle_pos

        self.stats.exclude("success")[env_ids] = 0.
        self.stats["success"][env_ids] = False

        if (env_ids == self.central_env_idx).any():
            self.payload_traj_vis.clear()
            self.drone_traj_vis.clear()
            self.draw.clear_lines()

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.drone_up = self.drone_state[..., 16:19]
        self.payload_pos, payload_rot = self.get_env_poses(self.payload.get_world_poses())
        self.payload_vels = self.payload.get_velocities()

        # relative position and heading
        self.drone_payload_rpos = self.drone_state[..., :3] - self.payload_pos.unsqueeze(1)
        self.target_payload_rpos = self.payload_target_pos - self.payload_pos.unsqueeze(1)
        obstacle_drone_rpos = self.obstacle_pos[..., [0, 2]] - self.drone_state[..., [0, 2]]

        obs = [
            self.drone_payload_rpos, # 3
            self.drone_state,
            self.target_payload_rpos, # 3
            self.payload_vels.unsqueeze(1), # 6
            obstacle_drone_rpos.flatten(start_dim=-2).unsqueeze(1), # 4
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)
        
        self.pos_error = torch.norm(self.target_payload_rpos, dim=-1)
        self.stats["pos_error"].lerp_(self.pos_error, (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))

        if self._should_render(0):
            central_env_pos = self.envs_positions[self.central_env_idx]
            drone_pos = (self.drone.pos[self.central_env_idx, 0]+central_env_pos).tolist()
            payload_pos = (self.payload_pos[self.central_env_idx]+central_env_pos).tolist()
            
            if len(self.payload_traj_vis)>1:
                point_list_0 = [self.payload_traj_vis[-1], self.drone_traj_vis[-1]]
                point_list_1 = [payload_pos, drone_pos]
                colors = [(1., .1, .1, 1.), (.1, 1., .1, 1.)]
                sizes = [1.5, 1.5]
                self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)
            
            self.drone_traj_vis.append(drone_pos)
            self.payload_traj_vis.append(payload_pos)
        
        return TensorDict({
            "agents": {
                "observation": obs,
            },
            "stats": self.stats
        }, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot, vels = self.drone_state[..., :13].split([3, 4, 6], dim=-1)
        
        # reward_pos = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
        reward_pos = 1 / (1 + self.reward_distance_scale * self.pos_error)

        bar_reward_up = normalize(-self.drone_payload_rpos)[..., 2]

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)

        spin = torch.square(vels[..., -1])
        reward_spin = 1. / (1.0 + torch.square(spin))

        swing = torch.norm(self.payload_vels[..., :3], dim=-1, keepdim=True)
        reward_swing = 1. * torch.exp(-swing)

        success = self.pos_error < 0.2

        collision = (
            self.obstacles
            .get_net_contact_forces()
            .any(-1)
            .any(-1, keepdim=True)
        )
        collision_reward = collision.float()

        self.stats["collision"].add_(collision_reward)
        assert bar_reward_up.shape == reward_spin.shape == reward_swing.shape
        reward = (
            reward_pos
            + reward_pos * (bar_reward_up + reward_spin + reward_swing) 
            # + success.float()
            + reward_effort
        ) * (1 - collision_reward)
        
        misbehave = (
            (pos[..., 2] < 0.2) 
            | (pos[..., 1].abs() > 2.)
            | (bar_reward_up < 0.1) 
            | (self.payload_pos[:, 2] > 3.).unsqueeze(-1)
        )
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        if self.reset_on_collision:
            terminated |= collision

        self.stats["success"].bitwise_or_(success)
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)
        
        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
