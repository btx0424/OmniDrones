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


import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D
from functorch import vmap

from omni.isaac.core.utils.viewports import set_camera_view

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, BinaryDiscreteTensorSpec

from omni.isaac.debug_draw import _debug_draw


class SDFNav(IsaacEnv):
    
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale

        super().__init__(cfg, headless)

        self.drone.initialize()
        self.capsules = RigidPrimView(
            "/World/envs/env_*/capsule_*",
            reset_xform_properties=False,
            track_contact_forces=True,
            shape=(-1, 10)
        )
        self.capsules.initialize()
        
        self.obstacle_pos = torch.zeros(self.num_envs, 10, 3, device=self.device)
        self.obstacle_pos_dist = D.Uniform(
            torch.tensor([-4, -4, 2.], device=self.device),
            torch.tensor([4, 4, 2.], device=self.device)
        )
        self.init_pos_dist = D.Uniform(
            torch.tensor([-2.5, -2.5, 1.], device=self.device),
            torch.tensor([2.5, 2.5, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        self.target_pos = torch.tensor([5.1, 5.1, 1.5], device=self.device)
        self.target_heading = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.alpha = 0.8

        self.scene_queries = torch.zeros(self.num_envs, 64, 3, device=self.device)
        self.last_pos_error = torch.zeros(self.num_envs, 1, device=self.device)

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        import omni_drones.utils.kit as kit_utils
        import omni.isaac.core.utils.prims as prim_utils
        from pxr import PhysxSchema, UsdPhysics

        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        self.drone.spawn([(0., 0., 0.5)])

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.8,
            improve_patch_friction=True,
        )

        for i in range(10):
            tree_prim = prim_utils.create_prim(
                f"/World/envs/env_0/capsule_{i}",
                prim_type="Capsule",
                attributes={"radius": 0.15, "height": 4.0},
                translation=(0, 0, 2),
            )
            UsdPhysics.RigidBodyAPI.Apply(tree_prim)
            UsdPhysics.CollisionAPI.Apply(tree_prim)
            # PhysxSchema.PhysxCollisionAPI.Apply(tree_prim)
            tree_prim.GetAttribute("physics:kinematicEnabled").Set(True)
        
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                    "scene": UnboundedContinuousTensorSpec((1, 64, 4), device=self.device),
                })
            }),
            "truncated": BinaryDiscreteTensorSpec(1, dtype=bool, device=self.device)
        }).expand(self.num_envs).to(self.device)
        
        action_dim = (
            self.drone.action_spec.shape[-1]
            + (3 + 3)
        )
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((1, action_dim), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "uprightness": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        
        # pos = self.init_pos_dist.sample((*env_ids.shape, 1))
        pos = torch.tensor([-5.1, -5.1, 1.5], device=self.device)
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        vel = torch.zeros((len(env_ids), 6), device=self.device)
        self.drone.set_velocities(vel, env_ids)

        obstacle_pos = self.obstacle_pos_dist.sample((len(env_ids), 10))
        self.obstacle_pos[env_ids] = obstacle_pos
        self.capsules.set_world_poses(
            obstacle_pos + self.envs_positions[env_ids].unsqueeze(1),
            env_indices=env_ids
        )
        self.scene_queries[env_ids] = torch.randn((len(env_ids), 64, 3), device=self.device)
        self.last_pos_error[env_ids] = (pos - self.target_pos).norm(dim=-1)
        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        control, query = tensordict[("agents", "action")].split([4, 6], dim=-1)
        self.effort = self.drone.apply_action(control)
        query_mean, query_std = query.split([3, 3], dim=-1)
        self.scene_queries[:] = (torch.randn((self.num_envs, 64, 3), device=self.device) + query_mean) * F.softplus(query_std)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]

        # relative position and heading
        self.rpos = self.target_pos - self.root_state[..., :3]
        self.rheading = self.target_heading - self.root_state[..., 13:16]
        
        obs = torch.cat([
            self.rpos, 
            self.root_state[..., 3:], 
            self.rheading,
        ], dim=-1)
        queries = self.drone.pos + self.scene_queries
        scene = torch.cat([
            queries,
            vmap(sdf_capsule)(queries, self.obstacle_pos, radius=0.15),
        ], dim=-1)

        if self._should_render(0):
            self.draw.clear_points()
            drone_pos = (self.drone.pos[self.central_env_idx] 
                         + self.envs_positions[self.central_env_idx]
                         ).cpu().squeeze(0)
            set_camera_view(
                eye=drone_pos.numpy() + np.asarray(self.cfg.viewer.eye),
                target=drone_pos.numpy() + np.asarray(self.cfg.viewer.lookat)                        
            )
            points = (self.scene_queries[self.central_env_idx].cpu() + drone_pos).tolist()
            self.draw.draw_points(points, [(0, 1, 0, 1)] * 64, [5] * 64)

        return TensorDict({
            "agents": {
                "observation": {
                    "state": obs,
                    "scene": scene.unsqueeze(1)
                }
            },
            "stats": self.stats,
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        # pose reward
        pos_error = torch.norm(self.rpos, dim=-1)
        reward_progress = self.last_pos_error - pos_error
        self.last_pos_error[:] = pos_error
        
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)

        reward = (
            5. * reward_progress 
            # + 1 / (1 + pos_error)
            + 0.2 * reward_up
        )
        
        sdf = vmap(sdf_capsule)(self.drone.pos, self.obstacle_pos, radius=0.15)
        done = (
            (sdf < 0.25).squeeze(1)
            | (self.drone.pos[..., 2] < 0.2)
            | (self.drone.pos[..., 2] > 2.5)
        )
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": done,
                "truncated": truncated
            },
            self.batch_size,
        )


def sdf_capsule(x: torch.Tensor, pos: torch.Tensor, radius: float):
    x = x[..., :2]
    pos = pos[..., :2]
    dist = torch.norm(x.unsqueeze(1) - pos.unsqueeze(0), dim=-1, keepdim=True) - radius
    return dist.min(1).values

