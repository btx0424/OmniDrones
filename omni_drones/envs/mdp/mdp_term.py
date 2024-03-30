import torch
import abc
from typing import Tuple
from omni_drones.envs.isaac_env import IsaacEnv
from omni_drones.robots.multirotor import Multirotor

RangeType = Tuple[float, float]

class MDPTerm():
    def __init__(self, env: IsaacEnv):
        self.env = env
    
    def update(self):
        pass

    def reset(self):
        pass
    
    def __call__(self) -> torch.Tensor:
        return self.compute()
    
    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def debug_vis(self):
        pass
    
    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device


class RewardFunc(MDPTerm):
    def __init__(self, env: IsaacEnv, weight: float=1.0):
        super().__init__(env)
        self.weight = weight
    
    def __call__(self):
        return self.weight * self.compute()
    

class TerminationFunc(MDPTerm):
    pass


class PositionOutOfRange(TerminationFunc):
    
    def __init__(
        self, 
        env: IsaacEnv, 
        x_range: RangeType = None,
        y_range: RangeType = None, 
        z_range: RangeType = None,
    ):
        super().__init__(env)
        self.robot: Multirotor = self.env.scene["drone"]
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
    
    def compute(self) -> torch.Tensor:
        terminated = torch.zeros(self.robot.shape, device=self.device, dtype=bool)
        root_pos_e = self.robot.data.root_pos_w - self.env.scene.env_origins
        for axis, _range in zip(range(3), (self.x_range, self.y_range, self.z_range)):
            if _range is not None:
                terminated.logical_or_(root_pos_e[..., axis] < _range[0])
                terminated.logical_or_(root_pos_e[..., axis] > _range[1])
        return terminated.unsqueeze(-1)


class TrackingErrorExceeds(TerminationFunc):
    def __init__(
        self,
        env: IsaacEnv,
        target_pos_key: str,
        thres: float
    ):
        super().__init__(env)
        self.thres = thres
        self.robot = self.env.drone
        self.target_pos_key = target_pos_key

    def compute(self) -> torch.Tensor:
        target_pos = getattr(self.env, self.target_pos_key)
        pos_diff = self.robot.root_pos_e - target_pos
        pos_error = pos_diff.norm(dim=-1, keepdim=True)
        terminated = (pos_error > self.thres)[:, 0]
        return terminated

