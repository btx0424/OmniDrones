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
import omni.isaac.core.utils.prims as prim_utils
import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.utils.torch import cpos, off_diag, others, make_cells, euler_to_quaternion
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from omni_drones.robots import ASSET_PATH


REGULAR_HEXAGON = [
    [0, 0, 0],
    [1.7321, -1, 0],
    [0, -2, 0],
    [-1.7321, -1, 0],
    [-1.7321, 1.0, 0],
    [0.0, 2.0, 0.0],
    [1.7321, 1.0, 0.0],
]

REGULAR_TETRAGON = [
    [0, 0, 0],
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [-1, 1, 0],
]

FORMATIONS = {
    "hexagon": REGULAR_HEXAGON,
    "tetragon": REGULAR_TETRAGON,
}

def sample_from_grid(cells: torch.Tensor, n):
    idx = torch.randperm(cells.shape[0], device=cells.device)[:n]
    return cells[idx]

class FormationFlyThrough(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.time_encoding = self.cfg.task.time_encoding
        self.safe_distance = self.cfg.task.safe_distance

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
            track_contact_forces=True
        )
        self.gate_frame.initialize()

        drone_state_dim = self.drone.state_spec.shape[0]
        obs_self_dim = drone_state_dim
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_self_dim += self.time_encoding_dim

        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)).to(self.device),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, 13+1)).to(self.device),
        })

        state_spec = CompositeSpec({
            "drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim)),
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
        self.cells = (
            make_cells([-3, -2, 0.5], [-1, 2, 2], [0.4, 0.4, 0.25])
            .flatten(0, -2)
            .to(self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([.2, .2, 0.], device=self.device) * torch.pi
        )
        self.target_pos = torch.tensor([2.0, 0.0, 1.5], device=self.device)
        self.formation = self.formation + self.target_pos
        self.target_heading = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_heading[..., 0] = -1
        self.crossed_plane = torch.zeros(self.num_envs, self.drone.n, device=self.device, dtype=bool)

        # additional infos & buffers
        stats_spec = CompositeSpec({
            # "cost_laplacian": UnboundedContinuousTensorSpec((self.num_envs, 1)),
            "cost_hausdorff": UnboundedContinuousTensorSpec((self.num_envs, 1)),
        }, shape=[self.num_envs]).to(self.device)
        self.observation_spec["stats"] = stats_spec

        self.stats = stats_spec.zero()

        # self.last_cost_l = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_cost_h = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_cost_pos = torch.zeros(self.num_envs, 1, device=self.device)

    def _design_scene(self) -> Optional[List[str]]:
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        scene_utils.design_scene()

        formation = self.cfg.task.formation
        if isinstance(formation, str):
            self.formation = torch.as_tensor(
                FORMATIONS[formation], device=self.device
            ).float()
        elif isinstance(formation, list):
            self.formation = torch.as_tensor(
                self.cfg.task.formation, device=self.device
            )
        else:
            raise ValueError(f"Invalid target formation {formation}")

        self.drone.spawn(translations=torch.zeros(len(self.formation), 3))

        prim_utils.create_prim(
            "/World/envs/env_0/Gate",
            usd_path=ASSET_PATH + "/usd/gate_sliding.usd",
            translation=(0., 0., 2.0),
        )

        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        
        drone_pos = vmap(sample_from_grid, randomness="different")(
            self.cells.expand(len(env_ids), *self.cells.shape), n=self.drone.n
        ) + self.envs_positions[env_ids].unsqueeze(1)
        drone_rpy = self.init_rpy_dist.sample((*env_ids.shape, self.drone.n))
        drone_rot = euler_to_quaternion(drone_rpy)
        vel = torch.zeros(len(env_ids), self.drone.n, 6, device=self.device)
        self.drone.set_world_poses(
            drone_pos, drone_rot[env_ids], env_ids
        )
        self.drone.set_velocities(vel, env_ids)

        self.last_cost_h[env_ids] = vmap(cost_formation_hausdorff)(
            drone_pos, desired_p=self.formation
        )
        com_pos = (drone_pos - self.envs_positions[env_ids].unsqueeze(1)).mean(1, keepdim=True)
        self.last_cost_pos[env_ids] = torch.square(
            com_pos - self.target_pos[env_ids]
        ).sum(2)

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.root_states = self.drone.get_state()
        pos = self.drone.pos
        self.root_states[..., :3] = self.target_pos - pos

        obs_self = [self.root_states]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs_self.append(t.expand(-1, self.drone.n, self.time_encoding_dim))
        obs_self = torch.cat(obs_self, dim=-1)

        relative_pos = vmap(cpos)(pos, pos)
        self.drone_pdist = vmap(off_diag)(torch.norm(relative_pos, dim=-1, keepdim=True))
        relative_pos = vmap(off_diag)(relative_pos)

        obs_others = torch.cat([
            relative_pos,
            self.drone_pdist,
            vmap(others)(self.root_states[..., 3:13])
        ], dim=-1)

        obs = TensorDict({
            "obs_self": obs_self.unsqueeze(2),
            "obs_others": obs_others,
        }, [self.num_envs, self.drone.n])

        state = TensorDict({"drones": self.root_states}, self.batch_size)

        return TensorDict({
            "drone.obs": obs, 
            "drone.state": state,
            "stats": self.stats,
        }, self.batch_size)

    def _compute_reward_and_done(self):
        crossed_plane = self.drone.pos[..., 0] > 0.
        crossing_plane = (crossed_plane & (~self.crossed_plane))
        self.crossed_plane |= crossed_plane
        distance_to_gate_plane = 0. - self.drone.pos[..., 0]
        distance_to_gate_center = torch.abs(self.drone.pos[..., 1:] - self.gate_pos[..., 1:])
        through_gate = (distance_to_gate_center < 0.5).all(-1)

        gate_reward = torch.where(
            distance_to_gate_plane > 0.,
            (0.4 - distance_to_gate_center).sum(-1) * torch.exp(-distance_to_gate_plane),
            1.
        )

        # cost_l = vmap(cost_formation_laplacian)(pos, desired_L=self.formation_L)
        cost_h = vmap(cost_formation_hausdorff)(self.drone.pos, desired_p=self.formation)
        
        cost_pos = torch.square(self.drone.pos.mean(-2, keepdim=True) - self.target_pos).sum(-1)

        reward_formation =  1 / (1 + torch.square(cost_h * 1.6)) 
        # reward_pos = 1 / (1 + cost_pos)

        # reward_formation = torch.exp(- cost_h * 1.6)
        reward_pos = torch.exp(- cost_pos)

        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values
        reward_separation = torch.square(separation / self.safe_distance).clamp(0, 1)
        reward = (
            reward_separation * (
                reward_formation 
                + reward_formation * reward_pos
                + 0.2 * reward_pos
            )
        ).unsqueeze(1).expand(-1, self.drone.n, 1)

        # self.last_cost_l[:] = cost_l
        self.last_cost_h[:] = cost_h
        self.last_cost_pos[:] = cost_pos

        self._tensordict["return"] += reward

        terminated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        crash = (pos[..., 2] < 0.2).any(-1, keepdim=True)

        done = terminated | crash | (separation<0.23)

        # self.stats["cost_laplacian"] -= cost_l
        self.stats["cost_hausdorff"] -= cost_h

        return TensorDict(
            {
                "reward": {"drone.reward": reward},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )


def cost_formation_laplacian(
    p: torch.Tensor,
    desired_L: torch.Tensor,
    normalized=False,
) -> torch.Tensor:
    """
    A scale and translation invariant formation similarity cost
    """
    L = laplacian(p, normalized)
    cost = torch.linalg.matrix_norm(desired_L - L)
    return cost.unsqueeze(-1)


def laplacian(p: torch.Tensor, normalize=False):
    """
    symmetric normalized laplacian

    p: (n, dim)
    """
    assert p.dim() == 2
    A = torch.cdist(p, p)
    D = torch.sum(A, dim=-1)
    if normalize:
        DD = D**-0.5
        A = torch.einsum("i,ij->ij", DD, A)
        A = torch.einsum("ij,j->ij", A, DD)
        L = torch.eye(p.shape[0], device=p.device) - A
    else:
        L = D - A
    return L


def cost_formation_hausdorff(p: torch.Tensor, desired_p: torch.Tensor) -> torch.Tensor:
    p = p - p.mean(-2, keepdim=True)
    desired_p = desired_p - desired_p.mean(-2, keepdim=True)
    cost = torch.max(directed_hausdorff(p, desired_p), directed_hausdorff(desired_p, p))
    return cost.unsqueeze(-1)


def directed_hausdorff(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    p: (*, n, dim)
    q: (*, m, dim)
    """
    d = torch.cdist(p, q, p=2).min(-1).values.max(-1).values
    return d
