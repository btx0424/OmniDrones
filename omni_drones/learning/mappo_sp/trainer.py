"""References:
    mlagents/trainers/ghost/trainer.py
    
    
    Roadmaps:
    - [] The simpliest case in which one team learns against another fiexed team
"""

from .actor import make_actor

from tensordict.nn import TensorDictModule
from torchrl.envs import TransformedEnv
from torch.cuda import _device_t
from typing import Union, Optional, Callable, Any

from dataclasses import dataclass


@dataclass
class SelfPlayParameters:
    window: int
    play_against_latest_model_ratio: float
    steps_between_save: int
    steps_between_swap: int
    steps_to_train_team: int


_create_actor_func_t = Callable[
    [],
    TensorDictModule,
]


class GhostTrainer:
    def __init__(
        self,
        env: TransformedEnv,
        device: _device_t,
        create_actor_func: Optional[_create_actor_func_t],
    ) -> None:
        self.device = device
        self.env = env

        self.actor = create_actor_func()
        self.actor = self.actor.to(self.device)
