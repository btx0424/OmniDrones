from .playerpool import Player, PlayerPool

from multielo import MultiElo

import torch
from tensordict import TensorDict
from omegaconf import DictConfig
from omni_drones.utils.torchrl.env import AgentSpec
from torch.cuda import _device_t
from tensordict.utils import NestedKey


class MAPPOSPPolicy:
    def __init__(
        self, cfg: DictConfig, agent_spec: AgentSpec, device: _device_t = "cuda"
    ) -> None:
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        self.clip_param: float = cfg.clip_param
        self.ppo_epoch: int = int(cfg.ppo_epochs)
        self.num_minibatches: int = int(cfg.num_minibatches)
        self.normalize_advantages: bool = cfg.get("normalize_advantages", False)

        self.entropy_coef: float = cfg.entropy_coef
        self.gae_gamma: float = cfg.gamma
        self.gae_lambda: float = cfg.gae_lambda

        self.action_dim: int = agent_spec.action_spec.shape[-1]
        
        self.reward_weights = torch.ones(
                self.agent_spec.reward_spec.shape, device=device
            )
        
        
        self.obs_name = ("agents", "observation")
        self.act_name = ("agents", "action")
        self.reward_name = ("agents", "reward")

    def train_op(self, tensordict: TensorDict):
        pass
