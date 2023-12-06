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
    CompositeSpec, 
    UnboundedContinuousTensorSpec, 
    DiscreteTensorSpec
)

import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
from omni.isaac.core.objects import DynamicCuboid

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
from omni_drones.utils.torch import euler_to_quaternion

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import cpos, off_diag, others, quat_axis
from omni_drones.robots.drone import MultirotorBase

from .utils import TransportationGroup, TransportationCfg
from ..utils import create_obstacle

class TransportFlyThrough(IsaacEnv):
    r"""
    A challenging cooperative control task where a group of UAVs carry a box-shaped payload connected via
    rigid links. 

    ## Observation
    The observation space is specified by a :py:class:`CompositeSpec` containing the following items:

    - `obs_self` (1, \*): The state of each UAV observed by itself, containing its kinematic
      information with the position being relative to the payload. It also includes a one-hot 
      vector indicating each drone's identity.
    - `obs_others` (k-1, \*): The observed states of other agents.
    - `obs_payload` (1, \*): The state of the frame, cotaining its position (relative to the
      reference), rotation (in quaternions and direction vectors), and velocities.
    - `obs_obstacles` (2, 2): Relative (x-z) positions of the obstacles.

    Reward
    ------
    - `seperation`: A factor that penalizes all agents when the minimum seperation is too small.
    - `pos`: Reward for tracking the trajectory, computed as :math:`\exp(-a * \text{pos_error})`.
    - `up`: Reward for keeping the payload upright.
    - `swing`: Reward for avoid large swinging of the payload.
    - `joint_limit`: Reward for kepping the joint states in a reasonalble range to avoid glitchy behaviors.

    ## Config
    - `obstacle_spacing` (Tuple[float, float], default=[0.85, 0.85]): A range from which the vertical spacing between the 
      obstacles is sampled.
    - `reset_on_collision` (bool, default=False): Whether to reset the environment when the payload collides with an 
      obstacle. 
    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.safe_distance = cfg.task.safe_distance
        self.obstacle_spacing = cfg.task.obstacle_spacing
        self.reset_on_collision = cfg.task.reset_on_collision
        self.collision_penalty = cfg.task.collision_penalty
        self.time_encoding = cfg.task.time_encoding

        super().__init__(cfg, headless)

        self.group.initialize(track_contact_forces=True)
        self.payload = self.group.payload_view

        self.obstacles = RigidPrimView(
            "/World/envs/env_*/obstacle_*",
            reset_xform_properties=False,
            shape=[self.num_envs, -1],
            track_contact_forces=True
        )
        self.obstacles.initialize()
        
        self.init_poses = self.group.get_world_poses(clone=True)
        self.init_velocities = torch.zeros_like(self.group.get_velocities())
        self.init_joint_pos = self.group.get_joint_positions(clone=True)
        self.init_joint_vel = torch.zeros_like(self.group.get_joint_velocities())
        self.obstacle_pos = self.get_env_poses(self.obstacles.get_world_poses())[0]

        self.init_drone_poses = self.drone.get_world_poses(clone=True)
        self.init_drone_vels = torch.zeros_like(self.drone.get_velocities())

        self.init_pos_dist = D.Uniform(
            torch.tensor([-2.5, -.5, 1.0], device=self.device),
            torch.tensor([-1.5, 0.5, 2.0], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([0., 0., -.5], device=self.device) * torch.pi,
            torch.tensor([0., 0., 0.5], device=self.device) * torch.pi
        )
        payload_mass_scale = self.cfg.task.payload_mass_scale
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(payload_mass_scale[0] * self.drone.MASS_0.sum(), device=self.device),
            torch.as_tensor(payload_mass_scale[1] * self.drone.MASS_0.sum(), device=self.device)
        )
        self.obstacle_spacing_dist = D.Uniform(
            torch.tensor(self.obstacle_spacing[0], device=self.device),
            torch.tensor(self.obstacle_spacing[1], device=self.device)
        )
        self.target_pos_dist = D.Uniform(
            torch.tensor([1.75, -.2, 1.25], device=self.device),
            torch.tensor([2.25, 0.2, 1.75], device=self.device)
        )
        self.payload_target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self.alpha = 0.8

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        group_cfg = TransportationCfg(num_drones=self.cfg.task.num_drones)
        self.group = TransportationGroup(drone=self.drone, cfg=group_cfg)

        scene_utils.design_scene()

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

        self.group.spawn(translations=[(0, 0, 2.0)], enable_collision=True)

        return ["/World/defaultGroundPlane"]
    
    def _set_specs(self):
        drone_obs_dim = self.drone.state_spec.shape[0] + self.drone.n
        payload_state_dim = 19
        if self.time_encoding:
            self.time_encoding_dim = 4
            payload_state_dim += self.time_encoding_dim

        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, drone_obs_dim)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, 13+1)),
            "obs_payload": UnboundedContinuousTensorSpec((1, payload_state_dim)),
            "obs_obstacles": UnboundedContinuousTensorSpec((2, 2)),
        }).to(self.device)

        state_spec = CompositeSpec({
            "drones": UnboundedContinuousTensorSpec((self.drone.n, drone_obs_dim)),
            "payload": UnboundedContinuousTensorSpec((1, payload_state_dim)),
            "obstacles": UnboundedContinuousTensorSpec((2, 2)),
        }).to(self.device)
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": observation_spec.expand(self.drone.n),
                "observation_central": state_spec,
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
        )

        info_spec = CompositeSpec({
            "payload_mass": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(self.drone.n),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "payload_pos_error": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "success": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "action_smoothness": UnboundedContinuousTensorSpec(self.drone.n),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["info"] = info_spec
        self.observation_spec["stats"] = stats_spec
        self.info = info_spec.zero()
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        pos = self.init_pos_dist.sample(env_ids.shape)
        rpy = self.init_rpy_dist.sample(env_ids.shape)
        rot = euler_to_quaternion(rpy)

        self.group._reset_idx(env_ids)
        self.group.set_world_poses(pos + self.envs_positions[env_ids], rot, env_ids)
        self.group.set_velocities(self.init_velocities[env_ids], env_ids)

        self.group.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.group.set_joint_velocities(self.init_joint_vel[env_ids], env_ids)

        payload_masses = self.payload_mass_dist.sample(env_ids.shape)
        self.info["payload_mass"][env_ids] = payload_masses.unsqueeze(-1).clone()
        self.payload.set_masses(payload_masses, env_ids)

        obstacle_spacing = self.obstacle_spacing_dist.sample(env_ids.shape)
        obstacle_pos = torch.zeros(len(env_ids), 2, 3, device=self.device)
        obstacle_pos[:, :, 2] = 1.2
        obstacle_pos[:, 1, 2] += obstacle_spacing
        self.obstacles.set_world_poses(
            obstacle_pos + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids
        )
        self.obstacle_pos[env_ids] = obstacle_pos
        self.payload_target_pos[env_ids] = self.target_pos_dist.sample(env_ids.shape)

        self.stats.exclude("success")[env_ids] = 0.
        self.stats["success"][env_ids] = False


    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        drone_pos = self.drone_states[..., :3]

        self.payload_pos, self.payload_rot = self.get_env_poses(self.payload.get_world_poses())
        payload_vels = self.payload.get_velocities()
        self.payload_heading: torch.Tensor = quat_axis(self.payload_rot, axis=0)
        self.payload_up: torch.Tensor = quat_axis(self.payload_rot, axis=2)
        
        self.drone_rpos = torch.vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = torch.vmap(off_diag)(self.drone_rpos)
        self.drone_pdist = torch.norm(self.drone_rpos, dim=-1, keepdim=True)
        payload_drone_rpos = self.payload_pos.unsqueeze(1) - drone_pos

        self.target_payload_rpos = self.payload_target_pos - self.payload_pos

        self.group.get_state()
        payload_state = [
            self.target_payload_rpos,
            self.payload_rot,  # 4
            payload_vels,  # 6
            self.payload_heading,  # 3
            self.payload_up, # 3
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            payload_state.append(t.expand(-1, self.time_encoding_dim))
        payload_state = torch.cat(payload_state, dim=-1).unsqueeze(1)

        obstacle_payload_rpos = self.obstacle_pos[..., [0, 2]] - self.payload_pos[..., [0, 2]].unsqueeze(1)

        obs = TensorDict({}, [self.num_envs, self.drone.n])
        identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)
        obs["obs_self"] = torch.cat(
            [-payload_drone_rpos, self.drone_states[..., 3:], identity], dim=-1
        ).unsqueeze(2) # [..., 1, state_dim]
        obs["obs_others"] = torch.cat(
            [self.drone_rpos, self.drone_pdist, torch.vmap(others)(self.drone_states[..., 3:13])], dim=-1
        ) # [..., n-1, state_dim + 1]
        obs["obs_payload"] = payload_state.expand(-1, self.drone.n, -1).unsqueeze(2) # [..., 1, 22]
        obs["obs_obstacles"] = obstacle_payload_rpos.unsqueeze(1).expand(-1, self.drone.n, 2, 2)

        state = TensorDict({}, self.num_envs)
        state["payload"] = payload_state # [..., 1, 22]
        state["drones"] = obs["obs_self"].squeeze(2) # [..., n, state_dim]
        state["obstacles"] = obstacle_payload_rpos # [..., 2, 2]

        self.payload_pos_error = torch.norm(self.target_payload_rpos, dim=-1, keepdim=True)
    
        self.stats["payload_pos_error"].lerp_(self.payload_pos_error, (1-self.alpha))
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
        joint_positions = (
            self.group.get_joint_positions()[..., :16]
            / self.group.joint_limits[..., :16, 0].abs()
        )
        
        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values

        reward = torch.zeros(self.num_envs, self.drone.n, 1, device=self.device)
        # reward_pose = 1 / (1 + torch.square(self.payload_pos_error * self.reward_distance_scale))
        reward_pose = torch.exp(-self.payload_pos_error * self.reward_distance_scale)
        reward_up = torch.square((self.payload_up[:, 2] + 1) / 2).unsqueeze(-1)

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort).mean(-1, keepdim=True)
        reward_separation = torch.square(separation / self.safe_distance).clamp(0, 1)
        reward_joint_limit = 0.5 * torch.mean(1 - torch.square(joint_positions), dim=-1)

        reward_action_smoothness = self.reward_action_smoothness_weight * -self.drone.throttle_difference

        collision = (
            self.obstacles
            .get_net_contact_forces()
            .any(-1)
            .any(-1, keepdim=True)
        )
        collision_reward = collision.float()
        self.stats["collision"].add_(collision_reward)

        reward[:] = (
            reward_separation * (
                reward_pose 
                + reward_pose * reward_up 
                + reward_joint_limit
                + reward_action_smoothness.mean(1, True)
                + reward_effort
            )  * (1. - self.collision_penalty * collision_reward)
        ).unsqueeze(1)

        done_misbehave = (
            (self.drone.pos[..., 2] < 0.2) 
            | (self.drone.pos[..., 2] > 3.6)
        )
        
        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1) 
            | done_misbehave.any(-1, keepdim=True)
            | (self.payload_pos[:, 2] < 0.3).unsqueeze(1)
        )

        if self.reset_on_collision:
            done = done | collision
        success = (self.payload_pos_error < 0.2) & (self.drone.pos[..., 0] > 0.05).all(-1, True)
        self.stats["success"].bitwise_or_(success)
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

