import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.transforms import CatTensors, ExcludeTransform
from torchrl.modules import ProbabilisticActor, NormalParamWrapper
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from .utils.gae import compute_gae
from .utils.valuenorm import ValueNorm1
from .modules.distributions import IndependentNormal

from omegaconf import DictConfig
from omni_drones.utils.torchrl.env import AgentSpec


from torch.cuda import _device_t
from typing import List, Union, Tuple,Sequence

from tensordict.utils import NestedKey


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = make_mlp([256, 128])
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
            ),
            nn.ELU(),
            nn.Flatten(),
            nn.LazyLinear(128),
        )
        self.value = nn.Sequential(nn.LazyLinear(128), nn.ELU(), nn.LazyLinear(1))

    def forward(self, obs_state: torch.Tensor, obs_map: torch.Tensor):
        feature_state = self.mlp(obs_state)
        feature_map = self.cnn(obs_map)
        feature = torch.cat([feature_state, feature_map], dim=-1)
        value = self.value(feature)
        return value


class PPOPingPong2v2(object):
    def __init__(
        self, cfg: DictConfig, agent_spec: AgentSpec, device: _device_t = "cuda"
    ):
        super().__init__()

        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        self.clip_param: float = cfg.clip_param
        self.ppo_epoch: int = int(cfg.ppo_epochs)
        self.num_minibatches: int = int(cfg.num_minibatches)
        self.normalize_advantages: bool = cfg.normalize_advantages

        self.entropy_coef: float = cfg.entropy_coef
        self.gae_gamma: float = cfg.gamma
        self.gae_lambda: float = cfg.gae_lambda

        self.act_dim: int = agent_spec.action_spec.shape[-1]

        self.reward_weights = torch.ones(
            self.agent_spec.reward_spec.shape, device=device
        )

        self.obs_name = ("agents", "observation")
        self.act_name = ("agents", "action")
        self.reward_name = ("agents", "reward")

        self.act_logps_name = f"{self.agent_spec.name}.action_logp"

        self.make_actor()
        self.make_critic()

        self.train_in_keys: List[NestedKey] = list(
            set(
                self.actor_in_keys
                + self.actor_out_keys
                + self.critic_in_keys
                + self.critic_out_keys
                + [
                    "next",
                    self.act_logps_name,
                    ("reward", self.reward_name),
                    "state_value",
                ]
                + ["progress", ("collector", "traj_ids")]
            )
        )

        self.n_updates: int = 0

    def make_actor(self):
        actor_cfg = self.cfg.actor

        self.actor = ProbabilisticActor(
            module=make_mlp(actor_cfg.hidden_units),
            in_keys=[self.obs_name, self.act_name],
            out_keys=[
                self.act_name,
                self.act_logps_name,
                f"{self.agent_spec.name}.action_entropy",
            ],
            distribution_class=D.Normal,
            return_log_prob=True,
            spec=self.agent_spec.action_spec,
        )

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_cfg.lr)


def make_mlp(num_units: Sequence[int]):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.ELU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


def make_actor(
    hidden_units: Sequence[int],
    actor_in_keys: List[NestedKey],
    actor_out_keys: List[NestedKey],
    action_spec: TensorSpec,
):
    module = NormalParamWrapper(make_mlp(hidden_units))
    td_module = TensorDictModule(
        module,
        in_keys=actor_in_keys,
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=td_module,
        in_keys=actor_in_keys,
        out_keys=actor_out_keys,
        spec=action_spec,
        return_log_prob=True,
        distribution_class=D.Normal,
    )
    return actor
