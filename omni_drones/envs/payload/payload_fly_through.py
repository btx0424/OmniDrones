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
    BinaryDiscreteTensorSpec,
    DiscreteTensorSpec
)

import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import omni.isaac.core.objects as objects
from omni.isaac.debug_draw import _debug_draw

import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView

from ..utils import create_obstacle
from .utils import attach_payload

class PayloadFlyThrough(IsaacEnv):
    r"""
    A challenging control task where the agent must fly the UAV with a payload through some obstacles.
    The vertical seperation between the obstacles is less than the connection length of the payload,
    such that the agent have to discover a way to swing the payload through the obstacles.

    ## Observation
    - `drone_payload_rpos` (3): The payload's postion relative to the drone.
    - `payload_vels` (6): The linear and angular velocities of the payload.
    - `target_payload_rpos` (3): The target payload position relative to the payload.
    - `root_state` (16 + num_rotors): The basic information of the drone (except its position), 
      containing its rotation (in quaternion), velocities (linear and angular), 
      heading and up vectors, and the current throttle.
    - `obstacle_drone_rpos` (2 * 2 = 4): The position of the two bars relative to the drone's position.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional
      vector encoding the current progress of the episode.
    
    ## Reward
    - `pos`: Reward for maintaining the final position of the payload around the target position. 
    - `up`: Reward for maintaining an upright orientation.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.
    - `swing`: Reward computed from the swing of the payload to discourage swinging.
    - `collision`: Reward for avoiding collisions with horizontal bars.

    The total reward is computed as follows: 

    ```{math}  
        r = [r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{spin} + r_\text{swing}) + r_\text{effort}] * (1 - r_\text{collision})
    ```

    ## Episode End

    The episode ends when the drone gets too close or too far to the ground, or when the payload gets too 
    close to the ground, or when the drone goes too far away horizontally, or when the maximum episode length 
    is reached, or (optional) when the drone collides with any obstacle.

    ## Config

    | Parameter               | Type                | Default       | Description |
    |-------------------------|---------------------|---------------|-------------|
    | `drone_model`           | str                 | "hummingbird" | Specifies the model of the drone being used in the environment. |
    | `reset_on_collision`    | bool                | False         | Indicates if the episode should reset when the drone collides with an obstacle. |
    | `bar_length`            | float               | 1.0           | Length of the pendulum's bar. |
    | `reward_distance_scale` | float               | 1.2           | Scales the reward based on the distance between the payload and its target. |
    | `time_encoding`         | bool                | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
    | `obstacle_spacing`      | tuple[float, float] | [0.85, 0.85]  | Specifies the minimum and maximum distance between two horizontal bars (obstacles) in the environment. |
 
    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.reset_on_collision = cfg.task.reset_on_collision
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
            reset_xform_properties=False,
        )
        self.payload.initialize()

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
            torch.tensor([-2.5, 0., 1.5], device=self.device),
            torch.tensor([-1.0, 0., 2.0], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([.2, .2, 2], device=self.device) * torch.pi
        )
        self.obstacle_spacing_dist = D.Uniform(
            torch.tensor(self.obstacle_spacing[0], device=self.device),
            torch.tensor(self.obstacle_spacing[1], device=self.device)
        )
        self.payload_target_pos_dist = D.Uniform(
            torch.tensor([1.3, 0., 1.0], device=self.device),
            torch.tensor([1.5, 0., 1.5], device=self.device)
        )
        payload_mass_scale = self.cfg.task.payload_mass_scale
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(payload_mass_scale[0] * self.drone.MASS_0, device=self.device),
            torch.as_tensor(payload_mass_scale[1] * self.drone.MASS_0, device=self.device)
        )

        self.payload_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.alpha = 0.8

        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.payload_traj_vis = []
        self.drone_traj_vis = []

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls()
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
        attach_payload(f"/World/envs/env_0/{self.drone.name}_0", self.cfg.task.bar_length)

        sphere = objects.DynamicSphere(
            "/World/envs/env_0/target",
            translation=(1.5, 0., 1.),
            radius=0.05,
            color=torch.tensor([1., 0., 0.])
        )
        kit_utils.set_collision_properties(sphere.prim_path, collision_enabled=False)
        kit_utils.set_rigid_body_properties(sphere.prim_path, disable_gravity=True)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 9 + 4
        if self.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((1, observation_dim))
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
            "payload_pos_error": UnboundedContinuousTensorSpec(1),
            "drone_uprightness": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "success": BinaryDiscreteTensorSpec(1, dtype=bool),
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

        payload_target_pos = self.payload_target_pos_dist.sample(env_ids.shape)
        self.payload_target_pos[env_ids] = payload_target_pos
        self.payload_target_vis.set_world_poses(
            payload_target_pos + self.envs_positions[env_ids], 
            env_indices=env_ids
        )
        payload_mass = self.payload_mass_dist.sample(env_ids.shape)
        self.payload.set_masses(payload_mass, env_ids)

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
        self.payload_pos = self.get_env_poses(self.payload.get_world_poses())[0]
        self.payload_vels = self.payload.get_velocities()

        # relative position and heading
        self.drone_payload_rpos = self.drone_state[..., :3] - self.payload_target_pos.unsqueeze(1)
        self.target_payload_rpos = (self.payload_target_pos - self.payload_pos).unsqueeze(1)
        obstacle_drone_rpos = self.obstacle_pos[..., [0, 2]] - self.drone_state[..., [0, 2]]
        
        obs = [
            self.drone_payload_rpos,
            self.drone_state[..., 3:],
            self.target_payload_rpos, # 3
            self.payload_vels.unsqueeze(1), # 6
            obstacle_drone_rpos.flatten(start_dim=-2).unsqueeze(1),
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        self.payload_pos_error = torch.norm(self.target_payload_rpos, dim=-1)
        self.stats["payload_pos_error"].lerp_(self.payload_pos_error, (1-self.alpha))
        self.stats["drone_uprightness"].lerp_(self.drone_up[..., 2], (1-self.alpha))

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
                "observation": obs
            },
            "stats": self.stats,
            # "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        
        reward_pos = 1.0 / (1.0 + torch.square(self.reward_distance_scale * self.payload_pos_error))
        # pose_reward = torch.exp(-distance * self.reward_distance_scale)

        reward_up = torch.square((self.drone_up[..., 2] + 1) / 2)

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)

        spin = torch.square(self.drone.vel[..., -1])
        reward_spin = 0.5 / (1.0 + torch.square(spin))

        swing = torch.norm(self.payload_vels[..., :3], dim=-1, keepdim=True)
        reward_swing = 0.5 * torch.exp(-swing)
        
        collision = (
            self.obstacles
            .get_net_contact_forces()
            .any(-1)
            .any(-1, keepdim=True)
        )
        collision_reward = collision.float()

        self.stats["collision"].add_(collision_reward)
        assert reward_pos.shape == reward_up.shape == reward_spin.shape == reward_swing.shape
        reward = (
            reward_pos 
            + reward_pos * (reward_up + reward_spin + reward_swing) 
            + reward_effort
        ) * (1 - collision_reward)
        
        misbehave = (
            (self.drone.pos[..., 2] < 0.2) 
            | (self.drone.pos[..., 2] > 2.5)
            | (self.drone.pos[..., 1].abs() > 2.)
            | (self.payload_pos[..., 2] < 0.15).unsqueeze(1)
        )
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        
        if self.reset_on_collision:
            terminated = terminated | collision

        done = terminated | truncated
        
        self.stats["success"].bitwise_or_(self.payload_pos_error < 0.2)
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
