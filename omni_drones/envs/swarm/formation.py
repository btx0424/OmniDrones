import torch
import functorch
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from tensordict.tensordict import TensorDict, TensorDictBase

import omni.isaac.core.utils.torch as torch_utils

from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec, Optional, List
from omni_drones.envs.utils.helpers import off_diag, cpos
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils

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

        observation_spec = CompositeSpec({
            "self": self.drone.state_spec.expand(
                1, *self.drone.state_spec.shape
            ).to(self.device),
            "others": UnboundedContinuousTensorSpec(
                (4, 3+3+1)
            ).to(self.device)
        })

        state_spec = CompositeSpec({
            "drones": self.drone.state_spec.expand(
                self.drone._count, *self.drone.state_spec.shape
            ),
        })

        self.agent_spec["drone"] = AgentSpec(
            "drone", 5,
            observation_spec,
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(3).to(self.device),
            state_spec.to(self.device)
        )
        self.ep_return = self._tensordict["return"]
        self._tensordict["return.cost_l"] = self.ep_return[..., 0]
        self._tensordict["return.cost_h"] = self.ep_return[..., 1]
        self._tensordict["return.height"] = self.ep_return[..., 2]

        self.last_cost_l = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_cost_h = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def DEFAULT_CAMERA_CONFIG(self):
        cfg = super().DEFAULT_CAMERA_CONFIG
        cfg.update({
            "parent_prim_path": f"/World/envs/env_0/{self.drone.name}_{0}",
            "translation": (2, 0, 3),
            "target": (0, 0, 0.7)
        })
        return cfg

    def _design_scene(self) -> Optional[List[str]]:
        cfg = RobotCfg()
        drone_cls = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        self.drone = drone_cls(cfg=cfg)

        scene_utils.design_scene()
        
        self.target_height = 1.5
        self.formation = torch.as_tensor(FORMATIONS["tetragon"], device=self.device).float()
        self.formation[:, 2] = self.target_height
        self.formation_L = laplacian(self.formation)

        self.drone.spawn(translations=self.formation)
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        _, rot = self.init_poses
        self.drone._reset_idx(env_ids)
        pos = torch.rand(len(env_ids), self.drone._count, 3, device=self.device)
        vel = torch.zeros(len(env_ids), self.drone._count, 6, device=self.device)
        self.drone.set_env_poses(pos, rot[env_ids], env_ids)
        self.drone.set_velocities(vel, env_ids)

        self.last_cost_h[env_ids] = functorch.vmap(cost_formation_laplacian)(pos, desired_L=self.formation_L)
        self.last_cost_l[env_ids] = functorch.vmap(cost_formation_hausdorff)(pos, desired_p=self.formation)
    
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict["drone.action"]
        self.effort = self.drone.apply_action(actions)
    
    def _compute_state_and_obs(self):
        states = self.drone.get_state()
        pos, vel = states[..., :3], states[..., 7:10]

        relative_pos = functorch.vmap(cpos)(pos, pos)
        relative_vel = functorch.vmap(cpos)(vel, vel)
        pdist = functorch.vmap(off_diag)(torch.norm(relative_pos, dim=-1, keepdim=True))
        relative_pos = functorch.vmap(off_diag)(relative_pos)
        relative_vel = functorch.vmap(off_diag)(relative_vel)

        obs = TensorDict({
            "self": states.unsqueeze(2),
            "others": torch.cat([relative_pos, relative_vel, pdist], dim=-1)
        }, [self.num_envs, self.drone._count])

        center = pos.mean(-2, keepdim=True)
        states[..., :3] -= center
        state = TensorDict({
            "drones": states
        }, self.batch_size)
        
        return TensorDict({
            "drone.obs": obs,
            "drone.state": state
        }, self.batch_size)
    
    def _compute_reward_and_done(self):
        pos, rot = self.drone.get_env_poses()

        reward = torch.zeros(self.num_envs, self.drone._count, 3, device=self.device)
        cost_l = functorch.vmap(cost_formation_laplacian)(pos, desired_L=self.formation_L)
        cost_h = functorch.vmap(cost_formation_hausdorff)(pos, desired_p=self.formation)
        cost_height = torch.square(pos[..., 2] - self.target_height)

        reward[..., 0] = self.last_cost_l - cost_l
        reward[..., 1] = self.last_cost_h - cost_h
        reward[..., 2] = - cost_height

        self.last_cost_l[:] = cost_l
        self.last_cost_h[:] = cost_h

        self._tensordict["return"] += reward
        done = (self.progress_buf >= self.max_eposode_length).unsqueeze(-1)
        
        return TensorDict({
            "reward": {
                "drone.reward": reward
            },
            "return": self._tensordict["return"],
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