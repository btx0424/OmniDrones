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
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec

import omni.isaac.core.objects as objects
import omni_drones.utils.kit as kit_utils
import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.debug_draw import _debug_draw

from omni_drones.utils.torch import euler_to_quaternion, normalize
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView

from ..utils import lemniscate, scale_time
from .utils import create_pendulum


class InvPendulumTrack(IsaacEnv):
    r"""
    An intermidiate control task where a drone-based inverted pendulum system
    need to track a reference lemniscate trajectory for the payload.

    ## Observation
    
    - `drone_payload_rpos` (3): The position of the drone relative to the payload's position.
    - `root_state` (16 + `num_rotors`): The basic information of the drone (except its position), 
      containing its rotation (in quaternion), velocities (linear and angular), 
      heading and up vectors, and the current throttle.
    - `target_payload_rpos` (3 * `future_traj_steps`): The position of the reference relative to the payload's position.
    - `payload_vel` (6): The linear and angular velocities of the payload.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional
      vector encoding the current progress of the episode.

    ## Reward
    
    - `pos`: Reward for tracking the trajectory based on how close the drone's payload is to the target position.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `action_smoothness`: Reward that encourages smoother drone actions, computed based on the throttle difference of the drone.

    The total reward is computed as follows:
    ```{math}
        r = r_\text{pos} + r_\text{effort} + r_\text{action_smoothness}
    ```

    ## Episode End
    The episode ends when the bar falls beyond a certain angle, or when the
    drone gets too close to the ground, or when the distance between the payload 
    and the target exceeds a threshold, or when the maximum episode length
    is reached.
    
    ## Config

    | Parameter               | Type  | Default   | Description |
    |-------------------------|-------|-----------|-------------|
    | `drone_model`           | str   | "firefly" | Specifies the model of the drone being used in the environment. |
    | `reset_thres`           | float | 0.5       | Threshold for the distance between the payload and its target, upon exceeding which the episode will be reset. |
    | `future_traj_steps`     | int   | 4         | Number of future trajectory steps the drone needs to predict. |
    | `reward_distance_scale` | float | 1.2       | Scales the reward based on the distance between the payload and its target. |
    | `time_encoding`         | bool  | True      | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |

    """
    def __init__(self, cfg, headless):
        self.reset_thres = cfg.task.reset_thres
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.future_traj_steps = int(cfg.task.future_traj_steps)
        self.bar_length = cfg.task.bar_length
        assert self.future_traj_steps > 0

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

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.1, -.1, 0.], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, 2.], device=self.device) * torch.pi
        )

        payload_mass_scale = self.cfg.task.payload_mass_scale
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(payload_mass_scale[0] * self.drone.MASS_0[0], device=self.device),
            torch.as_tensor(payload_mass_scale[1] * self.drone.MASS_0[0], device=self.device)
        )
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        self.traj_c_dist = D.Uniform(
            torch.tensor(-0.6, device=self.device),
            torch.tensor(0.6, device=self.device)
        )
        self.traj_scale_dist = D.Uniform(
            torch.tensor([1.8, 1.8, 1.], device=self.device),
            torch.tensor([3.2, 3.2, 1.5], device=self.device)
        )
        self.traj_w_dist = D.Uniform(
            torch.tensor(0.7, device=self.device),
            torch.tensor(1.0, device=self.device)
        )
        self.origin = torch.tensor([0., 0., 3.], device=self.device)
        self.traj_t0 = torch.pi / 2

        self.traj_c = torch.zeros(self.num_envs, device=self.device)
        self.traj_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.traj_w = torch.ones(self.num_envs, device=self.device)

        self.alpha = 0.8

        self.draw = _debug_draw.acquire_debug_draw_interface()

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
        create_pendulum(f"/World/envs/env_0/{self.drone.name}_0", self.cfg.task.bar_length, 0.04)

        return ["/World/defaultGroundPlane"]
    
    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3 * (self.future_traj_steps-1) + 9
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
            "tracking_error": UnboundedContinuousTensorSpec(1),
            "tracking_error_ema": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "payload_mass": UnboundedContinuousTensorSpec(1),
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        traj_w = self.traj_w_dist.sample(env_ids.shape)
        self.traj_w[env_ids] = torch.randn_like(traj_w).sign() * traj_w
        
        t0 = torch.zeros(len(env_ids), device=self.device)
        drone_pos = lemniscate(t0 + self.traj_t0, self.traj_c[env_ids]) + self.origin
        drone_pos[..., 2] -= self.bar_length
        drone_rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids], drone_rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.drone.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.drone.set_joint_velocities(self.init_joint_vels[env_ids], env_ids)

        payload_mass = self.payload_mass_dist.sample(env_ids.shape)
        self.payload.set_masses(payload_mass, env_ids)
        self.info["payload_mass"][env_ids] = payload_mass

        self.stats[env_ids] = 0.

        if self._should_render(0) and (env_ids == self.central_env_idx).any():
            # visualize the trajectory
            self.draw.clear_lines()

            traj_vis = self._compute_traj(self.max_episode_length, self.central_env_idx.unsqueeze(0))[0]
            traj_vis = traj_vis + self.envs_positions[self.central_env_idx]
            point_list_0 = traj_vis[:-1].tolist()
            point_list_1 = traj_vis[1:].tolist()
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [1 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.info["drone_state"][:] = self.drone_state[..., :13]
        payload_pos = self.get_env_poses(self.payload.get_world_poses())[0]
        self.payload_vels = self.payload.get_velocities()

        # relative position and heading
        target_pos = self._compute_traj(self.future_traj_steps, step_size=5)
        self.drone_payload_rpos = self.drone_state[..., :3] - payload_pos.unsqueeze(1)
        self.target_payload_rpos = target_pos - payload_pos.unsqueeze(1)

        obs = [
            self.drone_payload_rpos, # 3
            self.drone_state[..., 3:],
            self.target_payload_rpos.flatten(1).unsqueeze(1), # self.future_traj_steps * 3
            self.payload_vels.unsqueeze(1), # 6
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))

        return TensorDict({
            "agents": {
                "observation": obs,
            },
            "stats": self.stats,
            "info": self.info,
        }, self.batch_size)

    def _compute_reward_and_done(self):        

        distance = torch.norm(self.target_payload_rpos[:, [0]], dim=-1)
        self.stats["tracking_error"].add_(-distance)
        self.stats["tracking_error_ema"].lerp_(distance, (1-self.alpha))

        reward_pos = torch.exp(-self.reward_distance_scale * distance)

        reward_bar_up = normalize(-self.drone_payload_rpos)[..., 2]

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)
        
        reward = (
            reward_pos
            + reward_effort
            + reward_action_smoothness
        )
        
        done_misbehave = (self.drone.pos[..., 2] < 0.2) | (reward_bar_up < 0.2)
        done_hasnan = torch.isnan(self.drone_state).any(-1)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | done_misbehave
            | done_hasnan
            | (distance > self.reset_thres)
        )


        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["tracking_error"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": done,
            },
            self.batch_size,
        )
    
    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1.):
        if env_ids is None:
            env_ids = ...
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        t = self.traj_t0 + scale_time(self.traj_w[env_ids].unsqueeze(1) * t * self.dt)
        traj_rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, t.shape[1], 4)
        
        target_pos = vmap(lemniscate)(t, self.traj_c[env_ids])
        target_pos = vmap(torch_utils.quat_rotate)(traj_rot, target_pos) * self.traj_scale[env_ids].unsqueeze(1)

        return self.origin + target_pos

