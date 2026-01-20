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
from torchrl.data import Unbounded, Composite, DiscreteTensorSpec, BinaryDiscreteTensorSpec

import isaacsim.core.utils.torch as torch_utils
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.api.objects as objects
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView

from omni_drones.robots import ASSET_PATH

class FlyThrough(IsaacEnv):
    r"""
    A basic control task where the agent must fly the UAV through the gate.
    This task is a bit tricky because the gate moves horizontally in random locations.

    ## Observation

    - `drone_state` (16 + num_rotors): The basic information of the drone (except its position),
      containing its rotation (in quaternion), velocities (linear and angular),
      heading and up vectors, and the current throttle.
    - `target_drone_rpos` (3): The target position relative to the drone.
    - `gate_vel` (6): The linear and angular velocities of the gate.
    - `gate_drone_rpos` (2 * 2 = 4): The position of the gate relative to the drone's position.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional
      vector encoding the current progress of the episode.

    ## Reward

    - `pos`: Reward for maintaining the final position of the payload around the target position.
    - `gate`: Reward computed from the distance to the plane and the center of the gate.
    - `up`: Reward for maintaining an upright orientation.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.

    The total reward is computed as follows:

    ```{math}
        r = [r_\text{pos} + (0.5 * r_\text{gate}) + (r_\text{pos} + 0.3) * (r_\text{up} + r_\text{spin}) + r_\text{effort}]
    ```

    ## Episode End

    The episode ends when the drone gets too close or too far to the ground, or when the drone goes too
    far away horizontally, or when the drone gets too far to the gate, or when the drone passes by the gate, or when the maximum episode length
    is reached.

    ## Config

    | Parameter               | Type  | Default       | Description                                                                                                                                                                                                                             |
    | ----------------------- | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`           | str   | "Hummingbird" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `reset_on_collision`    | bool  | False         | Indicates if the episode should reset when the drone collides with an obstacle.                                                                                                                                                         |
    | `gate_moving_range`     | float | 1.0           | Moving range of the gate.                                                                                                                                                                                                               |
    | `gate_scale`            | float | 1.1           | Scale of the gate.                                                                                                                                                                                                                      |
    | `reward_distance_scale` | float | 1.0           | Scales the reward based on the distance between the payload and its target.                                                                                                                                                             |
    | `time_encoding`         | bool  | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.reset_on_collision = cfg.task.reset_on_collision
        self.gate_moving_range = cfg.task.gate_moving_range
        self.gate_scale = cfg.task.gate_scale
        super().__init__(cfg, headless)

        self.drone.initialize()

        self.gate = ArticulationView(
            "/World/envs/env_*/Gate",
            reset_xform_properties=False,
            shape=[self.num_envs, 1],
        )
        self.gate.initialize()
        self.gate_frame = RigidPrimView(
            "/World/envs/env_*/Gate/frame",
            reset_xform_properties=False,
            shape=[self.num_envs, 1],
            track_contact_forces=self.reset_on_collision
        )
        self.gate_frame.initialize()

        self.target = RigidPrimView(
            "/World/envs/env_*/target",
            reset_xform_properties=False,
        )
        self.target.initialize()

        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())
        self.obstacle_pos = self.get_env_poses(self.gate.get_world_poses())[0]
        self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.crossed_plane = torch.zeros(self.num_envs, 1, device=self.device, dtype=bool)

        self.init_pos_dist = D.Uniform(
            torch.tensor([-2.5, -1.5, 1.5], device=self.device),
            torch.tensor([-2.0, 1.5, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([.2, .2, 0.], device=self.device) * torch.pi
        )
        self.init_gate_pos_dist = D.Uniform(
            torch.tensor([-self.gate_moving_range], device=self.device),
            torch.tensor([self.gate_moving_range], device=self.device)
        )
        self.target_pos_dist = D.Uniform(
            torch.tensor([1.5, -1., 1.5], device=self.device),
            torch.tensor([2.5, 1., 2.5], device=self.device)
        )

        self.alpha = 0.7


    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        scale = torch.ones(3) * self.cfg.task.gate_scale
        prim_utils.create_prim(
            "/World/envs/env_0/Gate",
            usd_path=ASSET_PATH + "/usd/gate_sliding.usd",
            translation=(0., 0., 2.0),
            scale=scale
        )

        self.drone.spawn(translations=[(-2., 0.0, 2.0)])

        target = objects.DynamicSphere(
            prim_path="/World/envs/env_0/target",
            translation=(1.5, 0., 2.),
            radius=0.05,
            color=torch.tensor([1., 0., 0.])
        )
        kit_utils.set_collision_properties(target.prim_path, collision_enabled=False)
        kit_utils.set_rigid_body_properties(target.prim_path, disable_gravity=True)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 6
        if self.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim
        self.observation_spec = Composite({
            "agents": {
                "observation": Unbounded((1, observation_dim))
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = Composite({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = Composite({
            "agents": {
                "reward": Unbounded((1, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )
        stats_spec = Composite({
            "return": Unbounded(1),
            "episode_len": Unbounded(1),
            "pos_error": Unbounded(1),
            "drone_uprightness": Unbounded(1),
            "collision": Unbounded(1),
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

        self.crossed_plane[env_ids] = False

        target_pos = self.target_pos_dist.sample((*env_ids.shape, 1))
        self.target_pos[env_ids] = target_pos
        self.target.set_world_poses(
            target_pos + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids
        )
        gate_pos = self.init_gate_pos_dist.sample(env_ids.shape)
        self.gate.set_joint_positions(
            gate_pos, env_indices=env_ids
        )
        self.gate.set_joint_velocities(
            torch.zeros_like(gate_pos), env_indices=env_ids
        )
        self.gate.set_joint_position_targets(
            torch.zeros_like(gate_pos), env_indices=env_ids
        )

        self.stats.exclude("success")[env_ids] = 0.
        self.stats["success"][env_ids] = False

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.drone_up = self.drone_state[..., 16:19]
        self.gate_pos = self.get_env_poses(self.gate.get_world_poses())[0]
        self.gate_vel = self.gate.get_velocities()

        # relative position
        self.target_drone_rpos = self.target_pos - self.drone_state[..., :3]
        self.gate_drone_rpos = self.gate_pos - self.drone_state[..., :3]

        obs = [
            self.drone_state[..., 3:],
            self.target_drone_rpos,
            self.gate_vel[..., :3],
            self.gate_drone_rpos,
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        self.pos_error = torch.norm(self.target_drone_rpos, dim=-1)
        self.stats["pos_error"].mul_(self.alpha).add_((1-self.alpha) * self.pos_error)
        self.stats["drone_uprightness"].mul_(self.alpha).add_((1-self.alpha) * self.drone_up[..., 2])

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        crossed_plane = self.drone.pos[..., 0] > 0.
        crossing_plane = (crossed_plane & (~self.crossed_plane))
        self.crossed_plane |= crossed_plane
        distance_to_gate_plane = 0. - self.drone.pos[..., 0]
        distance_to_gate_center = torch.abs(self.drone.pos[..., 1:] - self.gate_pos[..., 1:])
        through_gate = (distance_to_gate_center < 0.5).all(-1)

        reward_gate = torch.where(
            distance_to_gate_plane > 0.,
            (0.4 - distance_to_gate_center).sum(-1) * torch.exp(-distance_to_gate_plane),
            1.
        )

        # pose reward
        distance_to_target = torch.norm(self.target_drone_rpos, dim=-1)

        # reward_pos = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance_to_target))
        reward_pos = torch.exp(-self.reward_distance_scale * distance_to_target)
        # uprightness
        reward_up = 0.5 * torch.square((self.drone_up[..., 2] + 1) / 2)

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)

        spin = torch.square(self.drone.vel[..., -1])
        reward_spin = 0.5 / (1.0 + torch.square(spin))

        if self.reset_on_collision:
            collision = (
                self.gate_frame
                .get_net_contact_forces()
                .any(-1)
                .any(-1, keepdim=True)
            )
            # collision_reward = collision.float()

            # self.stats["collision"].add_(collision_reward)
        assert reward_pos.shape == reward_up.shape == reward_spin.shape
        reward = (
            reward_pos
            + 0.5 * reward_gate
            + (reward_pos + 0.3) * (reward_up + reward_spin)
            + reward_effort
        ) # * (1 - collision_reward)

        misbehave = (
            (self.drone.pos[..., 2] < 0.2)
            | (self.drone.pos[..., 2] > 2.5)
            | (self.drone.pos[..., 1].abs() > 2.)
            | (distance_to_target > 6.)
        )
        hasnan = torch.isnan(self.drone_state).any(-1)
        invalid = (crossing_plane & ~through_gate)

        terminated = misbehave | hasnan | invalid
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        if self.reset_on_collision:
            terminated |= collision

        self.stats["success"].bitwise_or_(distance_to_target < 0.2)
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
