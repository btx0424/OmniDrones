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

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, normalize
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.objects as objects
import omni.isaac.core.materials as materials
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
)
from pxr import UsdShade
from omni.isaac.debug_draw import _debug_draw


class PingPongMultiV0(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)

        self.time_encoding = self.cfg.task.time_encoding

        self.drone.initialize()
        randomization = self.cfg.task.get("randomization", None)
        if randomization is not None:
            if "drone" in self.cfg.task.randomization:
                self.drone.setup_randomization(self.cfg.task.randomization["drone"])
        
        self.ball = RigidPrimView(
            "/World/envs/env_*/ball",
            reset_xform_properties=False,
            track_contact_forces=True,
            shape=(-1, 1)
        )
        self.ball.initialize()

        self.ball_mass_dist = D.Uniform(
            torch.tensor([0.04], device=self.device),
            torch.tensor([0.05], device=self.device)
        ) 
        self.init_ball_offset_dist = D.Uniform(
            torch.tensor([-0.3, -0.3, 1.6], device=self.device),
            torch.tensor([0.3, 0.3, 2.0], device=self.device)
        )
        self.init_drone_pos_dist = D.Uniform(
            torch.tensor([[0.5, -2.0, 1.5], [0.5, 0.5, 1.5]], device=self.device),
            torch.tensor([[-.5, -0.5, 1.7], [-.5, 2.0, 1.7]], device=self.device)
        )
        self.init_drone_rpy_dist = D.Uniform(
            torch.tensor([-.1, -.1, 0.], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, 2.], device=self.device) * torch.pi
        )

        self.turn = torch.zeros(self.num_envs, 2, dtype=bool, device=self.device)
        self.last_ball_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.alpha = 0.8
        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        material = materials.PhysicsMaterial(
            prim_path="/World/Physics_Materials/physics_material_0",
            restitution=0.8,
        )
        objects.DynamicSphere(
            prim_path="/World/envs/env_0/ball",
            radius=0.05,
            mass=0.05,
            color=torch.tensor([1., .2, .2]),
            physics_material=material,
        )
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        drone_prims = self.drone.spawn(translations=[(0.0, -1., 2.), (0.0, 1., 1.)])

        material = UsdShade.Material(material.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("base_link/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
        
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = (
            drone_state_dim 
            + 3 + 3 # ball's position and linear velocity
            + 3 + 6 # other's position and linear and angular velocity
            + 1 # turn
        )
        state_dim = (
            drone_state_dim * 2
            + 6 + 3 # relative ball pos, ball vel
            + 6 # relative drone pos, 
            + 2 # turn
        )

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim
            state_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((2, observation_dim)),
                "state": UnboundedContinuousTensorSpec(state_dim)
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": torch.stack([self.drone.action_spec]*self.drone.n, dim=0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((2, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 2,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state")
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "score": UnboundedContinuousTensorSpec(1),
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
        self.drone._reset_idx(env_ids, self.training)
        
        drone_pos = self.init_drone_pos_dist.sample(env_ids.shape)
        drone_rpy = self.init_drone_rpy_dist.sample((*env_ids.shape, 2))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(torch.zeros(len(env_ids), 2, 6, device=self.device), env_ids)

        turn = (
            torch.tensor([True, False], device=self.device)
            ^ (torch.randn(len(env_ids), 1, device=self.device) >= 0)
        )
        self.turn[env_ids] = turn
        ball_pos_offset = self.init_ball_offset_dist.sample(env_ids.shape)
        ball_pos = drone_pos[turn] + ball_pos_offset
        ball_rot = torch.tensor([1., 0., 0., 0.], device=self.device).repeat(len(env_ids), 1)
        self.ball.set_world_poses(
            ball_pos + self.envs_positions[env_ids], ball_rot, env_ids
        )
        self.ball.set_velocities(torch.zeros(len(env_ids), 6, device=self.device), env_ids)
        self.ball.set_masses(self.ball_mass_dist.sample(env_ids.shape), env_ids)
        self.last_ball_vel[env_ids] = 0.
        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]
        self.ball_pos, ball_rot = self.get_env_poses(self.ball.get_world_poses())
        self.ball_vel = self.ball.get_velocities()[..., :3]
        normalized_ball_vel = self.ball_vel / self.ball_vel.norm(dim=-1, keepdim=True).clip(1e-7)
        self.hit = (self.last_ball_vel * normalized_ball_vel).sum(dim=-1) < -0.1
        self.last_ball_vel[:] = normalized_ball_vel

        # relative position and heading
        self.rpos_ball =  self.ball_pos - self.drone.pos

        self.rpos_drone =  torch.stack([
            # [..., drone_id, [x, y, z]]
            self.drone.pos[..., 0, :] - self.drone.pos[..., 1, :],
            self.drone.pos[..., 1, :] - self.drone.pos[..., 0, :],
        ], dim=1)
        obs = [
            self.root_state, # [E, A, *]
            self.rpos_drone,
            self.drone.vel[:, [1, 0]],
            self.rpos_ball, # [E, A, 3] 
            self.ball_vel.expand(-1, 2, 3), # [E, 1, 3] -> [E, A, 3] ,
            self.turn.unsqueeze(-1),
        ]
        state = [
            self.root_state.flatten(-2),
            self.rpos_drone.flatten(-2),
            self.rpos_ball.flatten(-2),
            self.ball_vel.flatten(-2)
        ]
        
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs.append(t.expand(-1, 2, self.time_encoding_dim))
            state.append(t.squeeze(1))
        obs = torch.cat(obs, dim=-1)
        state = torch.cat(state, dim=-1)

        if self._should_render(0):
            central_env_pos = self.envs_positions[self.central_env_idx]
            drone_pos = self.drone.pos[self.central_env_idx]
            ball_pos = self.ball_pos[self.central_env_idx]
            turn = self.turn[self.central_env_idx]
            point_list_0 = (drone_pos[turn] + central_env_pos).tolist()
            point_list_1 = (ball_pos + central_env_pos).tolist()
            colors = [(1., 1., 0.5, .7)]
            sizes = [1.2]
            self.draw.clear_lines()
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

        return TensorDict({
            "agents": {
                "observation": obs,
            },
            "stats": self.stats,
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        
        hit = self.ball.get_net_contact_forces().any(-1)
        which_drone = self.rpos_ball.norm(dim=-1).argmin(dim=1, keepdim=True)
        switch_turn = hit & torch.take_along_dim(self.turn, which_drone, dim=1)
        self.turn.bitwise_xor_(torch.where(switch_turn, True, False))

        safety_cost = (1. - self.rpos_drone[..., 0, :].norm(dim=-1, keepdim=True)).clip(0.)
        reward_pos = self.turn.float() / (1 + torch.norm(self.rpos_ball[..., :2], dim=-1))
        reward_height = 0.8 * (
            self.ball_pos[..., 2] - self.drone.pos[..., 2].max(dim=1, keepdim=True)[0].clip(1.5)
        ).clip(0.3)
        reward_score = switch_turn.float() * 50.
        reward = torch.sum(
            reward_pos + reward_height + reward_score - safety_cost, 
            dim=1, keepdim=True
        ).expand(-1, 2)

        misbehave = (
            (self.drone.pos[..., 2] < 0.3).any(-1, keepdim=True)
            | (self.ball_pos[..., 2] < 0.2)
            | (self.ball_pos[..., 2] > 5.)
            | (self.ball_pos[..., 0].abs() > 2)
            | (self.ball_pos[..., 1].abs() > 2)
        ) # [E, 1]
        
        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | misbehave
        )

        self.stats["return"].add_(reward[..., [0]])
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["score"].add_(switch_turn.float())

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": done,
            },
            self.batch_size,
        )
