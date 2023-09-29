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

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.utils.torch import cpos, off_diag, others, euler_to_quaternion
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

def sample_from_grid(cells: torch.Tensor, n):
    idx = torch.randperm(cells.shape[0], device=cells.device)[:n]
    return cells[idx]

class Rearrange(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.safe_distance = self.cfg.task.safe_distance
        self.reward_share_ratio = self.cfg.task.reward_share_ratio

        self.drone.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)

        drone_state_dim = self.drone.state_spec.shape[0]
        observation_spec = CompositeSpec({
            "state_self": UnboundedContinuousTensorSpec((1, drone_state_dim)),
            "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, drone_state_dim+1)),
        }).to(self.device)

        state_spec = CompositeSpec({
            "drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim))
        }).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.drone.n,
            observation_spec,
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
            state_spec,
        )

        # initial state distribution
        # self.target_pos = torch.tensor(self.num_envs, self.drone.n, 3, device=self.device)
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )

        self.alpha = 0.7
        # additional infos & buffers
        stats_spec = CompositeSpec({
            "pos_error": UnboundedContinuousTensorSpec(self.drone.n, 1),
            "effort": UnboundedContinuousTensorSpec(self.drone.n, 1),
            # "collision": UnboundedContinuousTensorSpec(1)
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _design_scene(self) -> Optional[List[str]]:
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        scene_utils.design_scene()

        n = self.cfg.task.size
        x = torch.linspace(-n/2, n/2, n)
        y = torch.linspace(-n/2, n/2, n)
        zz = torch.ones(n, n) * 2
        translations = torch.stack([*torch.meshgrid(x, y), zz], dim=-1).flatten(0, 1)
        self.target_pos = translations.to(self.device)
        self.drone.spawn(translations=translations)

        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        
        pos_idx = torch.argsort(torch.rand(len(env_ids), self.drone.n), dim=-1)
        pos = self.target_pos[pos_idx]
        rpy = self.init_rpy_dist.sample((*env_ids.shape, self.drone.n))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )

        vel = torch.zeros(len(env_ids), self.drone.n, 6, device=self.device)
        self.drone.set_velocities(vel, env_ids)

        self.stats["pos_error"][env_ids] = 0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.drone_pos = self.drone_state[..., :3]
        self.drone_up = self.drone_state[..., 16:19]

        self.target_drone_rpos = self.target_pos - self.drone_pos
        drone_rpos = vmap(cpos)(self.drone_pos, self.drone_pos)
        drone_rpos = vmap(off_diag)(drone_rpos)
        self.drone_pdist = torch.norm(drone_rpos, dim=-1, keepdim=True)

        state_self = torch.cat([
            self.target_drone_rpos,
            self.drone_state[..., 3:],
        ], dim=-1)

        state_others = torch.cat([
            drone_rpos,
            self.drone_pdist,
            vmap(others)(self.drone_state[..., 3:])
        ], dim=-1)

        obs = TensorDict(
            {
                "state_self": state_self.unsqueeze(2),
                "state_others": state_others,
            },
            [self.num_envs, self.drone.n],
        )

        state = TensorDict({
            "drones": state_self,
        }, self.batch_size)

        self.stats["pos_error"].lerp_(self.target_drone_rpos.norm(dim=-1), 1-self.alpha)
        # self.stats
        return TensorDict({
            "drone.obs": obs, 
            "drone.state": state,
            "stats": self.stats
        }, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot = self.get_env_poses(self.drone.get_world_poses())

        distance = torch.norm(self.target_drone_rpos, dim=-1, keepdim=True)
        separation = self.drone_pdist.min(dim=-2).values
        spinnage = self.drone_state[..., 10:13].abs().sum(-1, keepdim=True)
        
        pos_reward = torch.exp(-distance)
        up_reward = torch.square((self.drone_up[..., 2].unsqueeze(-1) + 1) / 2)
        reward_spin = 1.0 / (1.0 + torch.square(spinnage))
        safety_reward = torch.square(separation / self.safe_distance).clamp(0, 1)

        reward = safety_reward * (
            pos_reward + pos_reward * (up_reward + spin_reward)
        )
        reward = reward.lerp(reward.mean(1, keepdim=True), self.reward_share_ratio)

        self._tensordict["return"] += reward

        terminated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        done_misbehave = ((pos[..., 2].unsqueeze(-1) < 0.2) | (distance > 5.0)).any(1)
        done_hasnan = torch.isnan(self.drone_state).any(-1).any(-1, keepdim=True)

        done = terminated | done_misbehave | done_hasnan

        return TensorDict(
            {
                "reward": {"drone.reward": reward},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )

