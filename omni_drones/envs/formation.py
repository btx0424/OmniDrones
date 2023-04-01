from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import torch

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.utils.torch import cpos, off_diag, others, make_cells
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
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

class Formation(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)

        drone_state_dim = self.drone.state_spec.shape[0]
        observation_spec = CompositeSpec({
            "self": UnboundedContinuousTensorSpec((1, drone_state_dim)).to(self.device),
            "others": UnboundedContinuousTensorSpec((self.drone.n-1, drone_state_dim+1)).to(self.device),
            "target": UnboundedContinuousTensorSpec((1, 3)).to(self.device)
        })

        state_spec = CompositeSpec(
            {
                "drones": self.drone.state_spec.expand(
                    self.drone.n, *self.drone.state_spec.shape
                ).to(self.device),
            }
        )

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            5,
            observation_spec,
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
            state_spec,
        )

        # initial state distribution
        self.cells = make_cells([-2, -2, 0.6], [2, 2, 2], [0.5, 0.5, 0.2], device=self.device).flatten(0, -2)
        self.target_pos = self.target_pos.expand(self.num_envs, 1, 3)
        
        # additional infos & buffers
        self.observation_spec["info"] = CompositeSpec({
            "cost_laplacian": UnboundedContinuousTensorSpec((self.num_envs, 1)),
            "cost_hausdorff": UnboundedContinuousTensorSpec((self.num_envs, 1)),
        }, shape=[self.num_envs]).to(self.device)
        self.metric_laplacian = torch.zeros(self.num_envs, 1, device=self.device)
        self.metric_hausdorff = torch.zeros(self.num_envs, 1, device=self.device)

        self.last_cost_l = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_cost_h = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_cost_pos = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def DEFAULT_CAMERA_CONFIG(self):
        cfg = super().DEFAULT_CAMERA_CONFIG
        cfg.update(
            {
                "parent_prim_path": "/World/envs/env_0",
                "translation": (2, 0, 3),
                "target": (0, 0, 0.7),
            }
        )
        return cfg

    def _design_scene(self) -> Optional[List[str]]:
        cfg = RobotCfg()
        drone_cls = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        self.drone: MultirotorBase = drone_cls(cfg=cfg)

        scene_utils.design_scene()

        self.target_pos = torch.tensor([0.0, 0.0, 1.5], device=self.device)
        
        if isinstance(self.cfg.task.formation, str):
            self.formation = torch.as_tensor(
                FORMATIONS["tetragon"], device=self.device
            ).float()
        elif isinstance(self.cfg.task.formation, list):
            self.formation = torch.as_tensor(
                self.cfg.task.formation, device=self.device
            )
        else:
            raise ValueError(f"Invalid target formation {self.cfg.task.formation}")

        self.formation = self.formation + self.target_pos
        self.formation_L = laplacian(self.formation)

        self.drone.spawn(translations=self.formation)
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        _, rot = self.init_poses
        self.drone._reset_idx(env_ids)
        
        pos = vmap(sample_from_grid, randomness="different")(
            self.cells.expand(len(env_ids), *self.cells.shape), n=self.drone.n
        ) + self.envs_positions[env_ids].unsqueeze(1)
        vel = torch.zeros(len(env_ids), self.drone.n, 6, device=self.device)
        self.drone.set_world_poses(pos, rot[env_ids], env_ids)
        self.drone.set_velocities(vel, env_ids)

        self.last_cost_h[env_ids] = vmap(cost_formation_laplacian)(
            pos, desired_L=self.formation_L
        )
        self.last_cost_l[env_ids] = vmap(cost_formation_hausdorff)(
            pos, desired_p=self.formation
        )
        com_pos = (pos - self.envs_positions[env_ids].unsqueeze(1)).mean(1, keepdim=True)
        self.last_cost_pos[env_ids] = torch.square(
            com_pos - self.target_pos[env_ids]
        ).sum(2)

        self.metric_laplacian[env_ids] = 0.
        self.metric_hausdorff[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        states = self.drone.get_state()
        pos = states[..., :3]
        states[..., :3] = self.target_pos - pos

        relative_pos = vmap(cpos)(pos, pos)
        self.pdist = vmap(off_diag)(torch.norm(relative_pos, dim=-1, keepdim=True))
        relative_pos = vmap(off_diag)(relative_pos)

        state_others = torch.cat([
            relative_pos,
            self.pdist,
            vmap(others)(states[..., 3:])
        ], dim=-1)

        obs = TensorDict(
            {
                "self": states.unsqueeze(2),
                "others": state_others,
                "target": states[..., :3].unsqueeze(2),
            },
            [self.num_envs, self.drone.n],
        )

        state = TensorDict({"drones": states}, self.batch_size)

        return TensorDict({
            "drone.obs": obs, 
            "drone.state": state,
        }, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot = self.get_env_poses(self.drone.get_world_poses())

        reward = torch.zeros(self.num_envs, self.drone.n, 1, device=self.device)
        cost_l = vmap(cost_formation_laplacian)(
            pos, desired_L=self.formation_L
        )
        cost_h = vmap(cost_formation_hausdorff)(pos, desired_p=self.formation)
        cost_pos = torch.square(pos.mean(-2, keepdim=True) - self.target_pos).sum(-1)

        reward_formation =  1 / (1 + torch.square(cost_h * 1.6)) 
        reward_pos = 1 / (1 + cost_pos)
        reward_separation = torch.square(self.pdist.min(dim=-2).values.min(dim=-2).values / 0.5).clamp(0, 1)
        reward[:] = reward_separation * (reward_formation + reward_formation * reward_pos).unsqueeze(-1)

        self.last_cost_l[:] = cost_l
        self.last_cost_h[:] = cost_h
        self.last_cost_pos[:] = cost_pos

        self._tensordict["return"] += reward

        terminated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        crash = (pos[..., 2] < 0.2).any(-1, keepdim=True)

        done = terminated | crash

        self.metric_laplacian -= cost_l
        self.metric_hausdorff -= cost_h

        return TensorDict(
            {
                "reward": {"drone.reward": reward},
                "return": self._tensordict["return"],
                "done": done,
                "info": {
                    "cost_laplacian": self.metric_hausdorff,
                    "cost_hausdorff": self.metric_hausdorff
                }
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
