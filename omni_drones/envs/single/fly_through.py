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
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, BinaryDiscreteTensorSpec

import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import omni.isaac.core.objects as objects
from pxr import UsdPhysics

import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.sensors.camera import Camera, PinholeCameraCfg
from omni_drones.views import ArticulationView, RigidPrimView

from omni_drones.robots import ASSET_PATH

class FlyThrough(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.reset_on_collision = self.cfg.task.reset_on_collision
        self.gate_moving_range = self.cfg.task.gate_moving_range
        self.gate_scale = self.cfg.task.gate_scale

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

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())
        self.obstacle_pos = self.get_env_poses(self.gate.get_world_poses())[0]
        self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.crossed_plane = torch.zeros(self.num_envs, 1, device=self.device, dtype=bool)

        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_spec = UnboundedContinuousTensorSpec(drone_state_dim + 6)

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            observation_spec.to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )

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

        stats_spec = CompositeSpec({
            "pos_error": UnboundedContinuousTensorSpec(1),
            "drone_uprightness": UnboundedContinuousTensorSpec(1),
            "success": BinaryDiscreteTensorSpec(1, dtype=bool)
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

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
        
        drone_prims = self.drone.spawn(translations=[(-2., 0.0, 2.0)])

        scale = torch.ones(3) * self.cfg.task.gate_scale
        prim_utils.create_prim(
            "/World/envs/env_0/Gate",
            usd_path=ASSET_PATH + "/usd/gate_sliding.usd",
            translation=(0., 0., 2.0),
            scale=scale
        )
        target = objects.DynamicSphere(
            "/World/envs/env_0/target",
            translation=(1.5, 0., 2.),
            radius=0.05,
            color=torch.tensor([1., 0., 0.])
        )
        kit_utils.set_collision_properties(target.prim_path, collision_enabled=False)
        kit_utils.set_rigid_body_properties(target.prim_path, disable_gravity=True)
        return ["/World/defaultGroundPlane"]

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

        self.stats["pos_error"][env_ids] = 0
        self.stats["drone_uprightness"][env_ids] = 0
        self.stats["success"][env_ids] = 0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.info["drone_state"][:] = self.drone_state
        self.drone_up = self.drone_state[..., 16:19]
        self.gate_pos, _ = self.get_env_poses(self.gate.get_world_poses())
        self.gate_vel = self.gate.get_velocities()

        # relative position
        self.target_drone_rpos = self.target_pos - self.drone_state[..., :3]
        self.gate_drone_rpos = self.gate_pos - self.drone_state[..., :3]

        state = torch.cat([
            self.drone_state[..., 3:],
            self.target_drone_rpos,
            self.gate_drone_rpos,
            self.gate_vel[..., :3],
        ], dim=-1)
        
        obs = state

        pos_error = torch.norm(self.target_drone_rpos, dim=-1)
        self.stats["pos_error"].mul_(self.alpha).add_((1-self.alpha) * pos_error)
        self.stats["drone_uprightness"].mul_(self.alpha).add_((1-self.alpha) * self.drone_up[..., 2])
        
        return TensorDict({
            "drone.obs": obs,
            "stats": self.stats,
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot, vels = self.drone_state[..., :13].split([3, 4, 6], dim=-1)
        
        crossed_plane = pos[..., 0] > 0.
        crossing_plane = (crossed_plane & (~self.crossed_plane))
        self.crossed_plane |= crossed_plane
        distance_to_gate_plane = 0. - pos[..., 0]
        distance_to_gate_center = torch.abs(pos[..., 1:] - self.gate_pos[..., 1:])
        through_gate = (distance_to_gate_center < 0.5).all(-1)

        reward_gate = torch.where(
            distance_to_gate_plane > 0.,
            (0.4 - distance_to_gate_center).sum(-1) * torch.exp(-distance_to_gate_plane),
            1.
        )

        # pose reward
        distance_to_target = torch.norm(self.target_drone_rpos, dim=-1)

        # reward_pose = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance_to_target))
        reward_pose = torch.exp(-self.reward_distance_scale * distance_to_target)
        # uprightness
        up_reward = .5 * torch.square((self.drone_up[..., 2] + 1) / 2)

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)

        spin = torch.square(vels[..., -1])
        reward_spin = .5 / (1.0 + torch.square(spin))

        assert reward_pose.shape == up_reward.shape == reward_spin.shape
        reward = (
            reward_pose 
            + 0.5 * reward_gate
            + (reward_pose + 0.3) * (up_reward + reward_spin) 
            + reward_effort
        )
        
        done_invalid = (crossing_plane & ~through_gate)
        done_misbehave: torch.Tensor = ((pos[..., 2] < 0.2) | (distance_to_target > 6.))
        if self.reset_on_collision:
            collision_frame = (
                self.gate_frame
                .get_net_contact_forces()
                .any(-1)
                .any(-1, keepdim=True)
            )
            done_misbehave.bitwise_or_(collision_frame)
        done_hasnan = torch.isnan(self.drone_state).any(-1)
        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | done_misbehave
            | done_invalid
            | done_hasnan
        )
        self.stats["success"].bitwise_or_(distance_to_target < 0.2)

        self._tensordict["return"] += reward.unsqueeze(-1)
        return TensorDict(
            {
                "reward": {"drone.reward": reward.unsqueeze(-1)},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )
