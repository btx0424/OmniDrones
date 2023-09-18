from typing import List
from .playerpool import Player, PlayerPool

from multielo import MultiElo

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from omegaconf import DictConfig
from omni_drones.utils.torchrl.env import AgentSpec
from torch.cuda import _device_t
from tensordict.utils import NestedKey


class SPPolicyWrapper:
    """
    The wrapper is used to capsulate two team policies so that they appear as one.
    Agreements:

    Each policy controls `num_players_in_team` agents.
    
    They are homogeneous.
    
    They are *only* used in DataCollector to generate trajectories.
    """

    def __init__(self, policies: List[TensorDictModule]) -> None:
        
        self.num_teams = len(policies)
        
        

        self.obs_name = ("agents", "observation")
        self.act_name = ("agents", "action")
        self.reward_name = ("agents", "reward")

    def __call__(self, in_tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError()
