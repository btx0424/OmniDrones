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
import omni.physx.scripts.utils as script_utils
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

import omni_drones.utils.kit as kit_utils
from pxr import UsdPhysics

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import cpos, off_diag, others
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.scene import design_scene
from omni_drones.utils.torch import euler_to_quaternion

from .utils import OveractuatedPlatform, PlatformCfg


def create_obstacle(
    prim_path: str,
    translation=(0., 0., 2.5),
    height: float=5,
):
    prim = prim_utils.create_prim(
        prim_path=prim_path,
        prim_type="Capsule",
        translation=translation,
        attributes={"radius":0.05, "height": height}
    )
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    kit_utils.set_collision_properties(
        prim_path, contact_offset=0.02, rest_offset=0
    )

    stage = prim_utils.get_current_stage()
    script_utils.createJoint(stage, "Fixed", prim.GetParent(), prim)
    return prim


class PlatformFlyThrough(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.reset_on_collision = self.cfg.task.reset_on_collision

        self.obstacles = RigidPrimView(
            "/World/envs/env_*/obstacle_*",
            reset_xform_properties=False,
            shape=[self.num_envs, -1],
            # track_contact_forces=True
        )
        self.obstacles.initialize()

        self.platform.initialize()
        
        self.init_vels = torch.zeros_like(self.platform.get_velocities())
        self.init_joint_pos = self.platform.get_joint_positions(clone=True)
        self.init_joint_vel = torch.zeros_like(self.platform.get_joint_velocities())
        self.obstacle_pos = self.get_env_poses(self.obstacles.get_world_poses())[0]

        drone_state_dim = self.drone.state_spec.shape.numel()
        observation_spec = CompositeSpec({
            "state_self": UnboundedContinuousTensorSpec((1, drone_state_dim + self.drone.n)),
            "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, drone_state_dim)),
            "state_frame": UnboundedContinuousTensorSpec((1, 19)),
            "obstacles": UnboundedContinuousTensorSpec((3, 2)),
        }).to(self.device)
        state_spec = CompositeSpec({
            "state_drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim + self.drone.n)),
            "state_frame": UnboundedContinuousTensorSpec((1, 19)),
            "obstacles": UnboundedContinuousTensorSpec((3, 2)),
        }).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.drone.n,
            observation_spec,
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
            state_spec
        )

        self.init_pos_dist = D.Uniform(
            torch.tensor([-2.5, -.5, 1.75], device=self.device),
            torch.tensor([-2.0, 0.5, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.4, -.4, 0.], device=self.device) * torch.pi,
            torch.tensor([0.4, 0.4, 2], device=self.device) * torch.pi
        )
        
        self.target_pos = torch.tensor([2.0, 0., 2.25], device=self.device)
        self.target_heading =  torch.zeros(self.num_envs, 3, device=self.device)
        self.target_up = torch.zeros(self.num_envs, 3, device=self.device)

        self.alpha = 0.7
        stats_spec = CompositeSpec({
            "pos_error": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "effort": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls()
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
            translations=[-1.5, 0., 2.],
            enable_collision=True
        )
        design_scene()

        obstacle_spacing = self.cfg.task.obstacle_spacing
        create_obstacle("/World/envs/env_0/obstacle_0", translation=(0.,-obstacle_spacing, 2.5))
        create_obstacle("/World/envs/env_0/obstacle_1", translation=(0., 0., 2.5))
        create_obstacle("/World/envs/env_0/obstacle_2", translation=(0., obstacle_spacing, 2.5))
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        platform_pos = self.init_pos_dist.sample(env_ids.shape) + self.envs_positions[env_ids]
        platform_rpy = self.init_rpy_dist.sample(env_ids.shape)
        platform_rot = euler_to_quaternion(platform_rpy)
        self.platform.set_world_poses(platform_pos, platform_rot, env_indices=env_ids)
        self.platform.set_velocities(self.init_vels[env_ids], env_ids)

        self.platform.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.platform.set_joint_velocities(self.init_joint_vel[env_ids], env_ids)

        self.stats["pos_error"][env_ids] = 0
        self.stats["collision"][env_ids] = 0
        self.stats["effort"][env_ids] = 0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        drone_pos = self.drone_states[..., :3]
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)

        self.platform_pos, platform_rot = self.get_env_poses(self.platform.get_world_poses(clone=True))
        self.platform_heading = vmap(torch_utils.quat_axis)(platform_rot, axis=0)
        self.platform_up = vmap(torch_utils.quat_axis)(platform_rot, axis=2)
        self.platform_vels = self.platform.get_velocities(clone=True)

        self.target_platform_rpos = self.target_pos - self.platform_pos

        platform_drone_rpos = self.platform_pos - self.drone_states[..., :3]
        platform_state = torch.cat([
            self.target_platform_rpos, # 3
            platform_rot, # 4
            self.platform_heading, # 3
            self.platform_up, # 3
            self.platform_vels, # 6
        ], dim=-1) # [num_envs, 1, 19]

        obstacle_platform_rpos = self.obstacle_pos[..., [0, 1]] - self.platform_pos[..., [0, 1]]

        identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)

        obs = TensorDict({}, [self.num_envs, self.drone.n])
        obs["state_self"] = torch.cat(
            [-platform_drone_rpos, self.drone_states[..., 3:], identity], dim=-1
        ).unsqueeze(2)
        obs["state_others"] = torch.cat(
            [self.drone_rpos, vmap(others)(self.drone_states[..., 3:])], dim=-1
        )
        obs["state_frame"] = platform_state.unsqueeze(1).expand(-1, self.drone.n, 1, -1)
        obs["obstacles"] = obstacle_platform_rpos.unsqueeze(1).expand(-1, self.drone.n, 3, 2)

        state = TensorDict({}, [self.num_envs])
        state["state_drones"] = obs["state_self"].squeeze(2)    # [num_envs, drone.n, drone_state_dim]
        state["state_frame"] = platform_state                # [num_envs, 1, platform_state_dim]
        state["obstacles"] = obstacle_platform_rpos    # [num_envs, 3, 2]
        
        pos_error = torch.norm(self.target_platform_rpos, dim=-1)
        self.stats["pos_error"].mul_(self.alpha).add_((1-self.alpha) * pos_error)
        return TensorDict(
            {
                "drone.obs": obs,
                "drone.state": state,
                "stats": self.stats
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        distance = torch.norm(self.target_platform_rpos, dim=-1)
        
        reward = torch.zeros(self.num_envs, self.drone.n, 1, device=self.device)
        reward_pose = 1 / (1 + torch.square(distance * self.reward_distance_scale))
        
        spinnage = self.platform_vels[:, -3:].abs().sum(-1)
        reward_spin = 1. / (1 + torch.square(spinnage))
        
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort).mean(-1, keepdim=True)

        # collision = (
        #     self.obstacles
        #     .get_net_contact_forces()
        #     .any(-1)
        #     .any(-1, keepdim=True)
        # )
        # collision_reward = collision.float()
        # self.stats["collision"] += collision_reward

        reward[:] = (
            (
                reward_pose 
                + reward_pose * (reward_spin) 
                + reward_effort
            ) 
        ).unsqueeze(1)

        done_misbehave = (
            (self.drone_states[..., 2] < 0.25).any(-1, keepdim=True)
            | (self.platform_pos[..., 1].abs() > 1.5)
        )
        done_hasnan = torch.isnan(self.drone_states).any(-1)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | done_misbehave
            | done_hasnan.any(-1, keepdim=True)
        )
        # if self.reset_on_collision:
        #     done |= collision

        self._tensordict["return"] += reward
        return TensorDict(
            {
                "reward": {"drone.reward": reward},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )

