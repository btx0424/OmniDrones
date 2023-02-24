import torch
import functorch
from torchrl.data import UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict, TensorDictBase

import omni.isaac.core.utils.torch as torch_utils

from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec, Optional, List
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import (
    Crazyflie, Firefly, Neo11, Hummingbird
)
import omni_drones.utils.kit as kit_utils

REGULAR_HEXAGON = [
    [0, 0, 0],
    [1.7321, -1, 0],
    [0, -2, 0],
    [-1.7321, -1, 0],
    [-1.7321, 1., 0],
    [0., 2., 0.],
    [1.7321, 1., 0.]
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

class Formation(IsaacEnv):

    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.init_poses = self.drone.get_env_poses(clone=True)
        observaton_spec = ...
        self.agent_spec["drone"] = AgentSpec(
            "drone", 5,
            self.drone.state_spec,
            self.drone.action_spec,
            UnboundedContinuousTensorSpec(3).to(self.device)
        )

    def _design_scene(self) -> Optional[List[str]]:
        cfg = RobotCfg()
        self.drone = Firefly(cfg=cfg)
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.formation = torch.as_tensor(FORMATIONS["tetragon"], device=self.device).float()
        self.formation_L = laplacian(self.formation)
        self.target_height = 1.0
        self.drone.spawn(5, translation=self.formation)
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        _, rot = self.init_poses
        self.drone._reset_idx(env_ids)
        pos = torch.rand(len(env_ids), 1, 3, device=self.device)
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        self.drone.set_env_poses(pos, rot[env_ids], env_ids)
        self.drone.set_velocities(vel, env_ids)
    
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict["drone.action"]
        self.effort = self.drone.apply_action(actions)
    
    def _compute_state_and_obs(self):
        obs = self.drone.get_state()
        return TensorDict({
            "drone.obs": obs,
        }, self.batch_size)
    
    def _compute_reward_and_done(self):
        pos, rot = self.drone.get_env_poses()

        reward = torch.zeros(self.num_envs, self.drone._count, 3)
        cost_f1 = cost_formation_laplacian(pos, self.formation_L)
        cost_f2 = cost_formation_hausdorff(pos, self.formation)
        cost_height = torch.square(pos[..., 2] - self.target_height)

        reward[..., 0] = - cost_f1
        reward[..., 1] = - cost_f2
        reward[..., 2] = - cost_height

        done = (self.progress_buf >= self.max_eposode_length).unsqueeze(-1)
        return TensorDict({
            "reward": {
                "drone.reward": reward
            },
            "done": done
        }, self.batch_size)

def cost_formation_laplacian(
    p: torch.Tensor, desired_L: torch.Tensor, normalized=False,
) -> torch.Tensor:
    """
        A scale and translation invariant formation similarity cost
    """
    L = laplacian(p, normalized)
    cost = torch.linalg.matrix_norm(desired_L-L)
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
        DD = D ** -0.5
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