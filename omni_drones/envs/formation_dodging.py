from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core import objects
import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.utils.torch import cpos, off_diag, others, make_cells, euler_to_quaternion
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

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

class FormationDodge(IsaacEnv):
    def __init__(self, cfg, headless):
        self.time_encoding = cfg.task.time_encoding
        self.safe_distance = cfg.task.safe_distance
        self.map_size = 20

        super().__init__(cfg, headless)
        # self.ball = RigidPrimView(
        #     "/World/envs/env_*/ball_*",
        #     reset_xform_properties=False,
        #     shape=(-1, 3)
        # )
        # self.ball.initialize()
        self.drone.initialize()

        # initial state distribution
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        self.target_pos = self.target_pos.expand(self.num_envs, 1, 3)
        self.target_heading = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_heading[..., 0] = -1
        
        self.i0 = (
            torch.arange(self.num_envs, device=self.device)
            .view(self.num_envs, 1)
            .expand(self.num_envs, self.drone.n)
        )
        self.i1 = (
            torch.arange(self.drone.n, device=self.device)
            .view(1, self.drone.n)
            .expand(self.num_envs, self.drone.n)
        )
        self.minimap = torch.zeros(
            (self.num_envs, self.drone.n, self.map_size, self.map_size), device=self.device)

        self.alpha = 0.8

    def _design_scene(self) -> Optional[List[str]]:
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        scene_utils.design_scene()

        self.target_pos = torch.tensor([0.0, 0.0, 1.5], device=self.device)
        
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

        self.formation = self.formation + self.target_pos
        # for i in range(3):
        #     objects.DynamicSphere(
        #         f"/World/envs/env_0/ball_{i}",
        #         radius=0.04,
        #     )

        self.drone.spawn(translations=self.formation)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_obs_dim = (
            self.drone.state_spec.shape[0] 
            + 3 # relative pos to the target
            + self.drone.n # identity
        )
        if self.time_encoding:
            self.time_encoding_dim = 4
            drone_obs_dim += self.time_encoding_dim

        # observation_spec = CompositeSpec({
        #     "obs_self": UnboundedContinuousTensorSpec(drone_obs_dim),
        #     "obs_minimap": UnboundedContinuousTensorSpec((2, self.map_size, self.map_size))
        # }).to(self.device)

        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, drone_obs_dim)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, 14))
        }).to(self.device)

        state_spec = CompositeSpec({
            "drones": UnboundedContinuousTensorSpec((self.drone.n, drone_obs_dim)),
        }).to(self.device)

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": observation_spec.expand(self.drone.n),
            },
            "state": state_spec
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": torch.stack([self.drone.action_spec]*self.drone.n, dim=0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((self.drone.n, 1))
            }
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key="state"
        )

        # additional infos & buffers
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(self.drone.n),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "cost_hausdorff": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1)
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        
        pos = self.envs_positions[env_ids].unsqueeze(1) + self.formation
        rpy = self.init_rpy_dist.sample((*env_ids.shape, self.drone.n))
        rot = euler_to_quaternion(rpy)
        vel = torch.zeros(len(env_ids), self.drone.n, 6, device=self.device)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(vel, env_ids)

        # theta = torch.rand((len(env_ids), 3), device=self.device) * 2 * torch.pi
        # r = torch.rand((len(env_ids), 3), device=self.device) + 5. 
        # ball_pos = (
        #     torch.stack([torch.cos(theta), torch.sin(theta), torch.ones_like(theta)], dim=-1)
        #     * torch.stack([r, r, torch.ones_like(r)], dim=1)
        # )
        # ball_vel = torch.zeros(len(env_ids), 3, 6, device=self.device)
        # self.ball.set_world_poses(ball_pos, env_indices=env_ids)
        # self.ball.set_velocities(ball_vel, env_indices=env_ids)

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.root_states = self.drone.get_state()

        identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)
        obs_self = [self.root_states, self.target_pos - self.drone.pos, identity]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs_self.append(t.expand(-1, self.drone.n, self.time_encoding_dim))
        obs_self = torch.cat(obs_self, dim=-1)

        i, j = ((self.drone.pos[..., :2] + 2.5) / 5 * self.map_size).long().unbind(-1)
        self.minimap[self.i0, self.i1, i, j] = 1.
        minimap = torch.cat([
            self.minimap.unsqueeze(2), 
            self.minimap.unsqueeze(2).sum(1, keepdim=True).expand(-1, self.drone.n, 1, -1, -1)
        ], dim=2) # [num_envs, drone.n, 2, map_size, map_size]

        relative_pos = vmap(cpos)(self.drone.pos, self.drone.pos)
        relative_pos = vmap(off_diag)(relative_pos)
        self.drone_pdist = torch.norm(relative_pos, dim=-1, keepdim=True)

        obs_others = torch.cat([
            relative_pos,
            self.drone_pdist,
            vmap(others)(self.root_states[..., 3:13])
        ], dim=-1)

        # ball_pos = self.get_env_poses(self.ball.get_world_poses())[0]
        # ball_vel = self.ball.get_velocities()[..., :3]
        # obs_ball = torch.cat([ball_pos, ball_vel], dim=-1)

        obs = TensorDict({
            "obs_self": obs_self.unsqueeze(2),
            "obs_others": obs_others,
            # "obs_ball": obs_ball.unsqueeze(1).expand(-1, self.drone.n, -1, -1),
            # "mask": ...
        }, [self.num_envs, self.drone.n])

        # obs = TensorDict({
        #     "obs_self": obs_self,
        #     "obs_minimap": minimap,
        #     # "obs_ball": obs_ball.unsqueeze(1).expand(-1, self.drone.n, -1, -1),
        #     # "mask": ...
        # }, [self.num_envs, self.drone.n])

        state = TensorDict({
            "drones": obs_self,
            # "balls": obs_ball,
            # "mask": ...
        }, self.batch_size)

        return TensorDict({
            "agents": {
                "observation": obs, 
            },
            "state": state,
            "stats": self.stats
        }, self.batch_size)

    def _compute_reward_and_done(self):

        cost_h = vmap(cost_formation_hausdorff)(self.drone.pos, desired_p=self.formation)
        
        distance = torch.norm(self.drone.pos.mean(-2, keepdim=True) - self.target_pos, dim=-1)

        # reward_formation =  1 / (1 + torch.square(cost_h * 1.6)) 
        reward_formation =  torch.exp(- cost_h * 1.6) 
    
        reward_pos = torch.exp(- distance)
        reward_heading = self.drone.heading[..., 0].mean(-1, True)

        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values
        reward_separation = torch.square(separation / self.safe_distance).clamp(0, 1)
        reward = (
            reward_separation * (
                reward_formation 
                + reward_formation * (reward_pos + reward_heading)
                + 0.4 * reward_pos
            )
        ).expand(-1, self.drone.n)

        terminated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        crash = (self.drone.pos[..., 2] < 0.2).any(-1, keepdim=True)
        # out = (self.drone.pos[..., :2].abs() > 2.2).any(dim=-1).any(-1, keepdim=True)

        done = terminated | crash | (separation<0.23) 

        self.stats["cost_hausdorff"].lerp_(cost_h, (1-self.alpha))
        self.stats["pos_error"].lerp_(distance, (1-self.alpha))
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
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
