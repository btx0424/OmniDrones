import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

from ..utils.gae import compute_gae
from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal

@dataclass
class PPOConfig:
    name: str = "ppo_tconv"
    train_every: int = 64
    ppo_epochs: int = 4
    num_minibatches: int = 16

    condition_mode: str = "cat"


cs = ConfigStore.instance()
cs.store("ppo_tconv", node=PPOConfig, group="algo")


def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale


class TConv(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.tconv = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=1), nn.ELU(),
            nn.LazyConv1d(64, kernel_size=7, stride=2), nn.ELU(),
            nn.LazyConv1d(64, kernel_size=5, stride=2), nn.ELU(),
        )
        self.mlp = make_mlp([256, 256])
        self.out = nn.LazyLinear(out_dim)
    
    def forward(self, features: torch.Tensor):
        batch_shape = features.shape[:-2]
        features = features.flatten(0, -3) # [*, D, T]
        features_tconv = self.tconv(features).flatten(1)
        features = torch.cat([features_tconv, features[:, :, -1]], dim=1)
        features = self.mlp(features)
        features = self.out(features)
        return features.unflatten(0, batch_shape)


class PPOTConvPolicy:

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device
    ):
        self.cfg = cfg
        self.device = device

        self.entropy_coef = 0.001
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec.shape[-2:]

        fake_input = observation_spec.zero()
        
        self.history_encoder = TensorDictModule(
            TConv(128), [("agents", "observation_h")], ["context"]
        ).to(self.device)

        if self.cfg.condition_mode == "cat":
            condition = lambda: CatTensors(["_feature", "context"], "_feature", del_keys=False)
        elif self.cfg.condition_mode == "film":
            condition = lambda: TensorDictModule(FiLM(128), ["_feature", "context"], ["_feature"])

        actor = TensorDictSequential(
            TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["_feature"]),
            condition(),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 256]), Actor(self.action_dim)), 
                ["_feature"], ["loc", "scale"]
            )
        )
        
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        self.critic = TensorDictSequential(
            TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["_feature"]),
            condition(),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)), 
                ["_feature"], ["state_value"]
            )
        ).to(self.device)

        self.history_encoder(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        self.encoder_opt = torch.optim.Adam(self.history_encoder.parameters(), lr=5e-4)
        self.value_norm = ValueNorm1(reward_spec.shape[-2:]).to(self.device)
    
    def __call__(self, tensordict: TensorDict):
        tensordict = self.history_encoder(tensordict)
        tensordict = self.actor(tensordict)
        tensordict = self.critic(tensordict)
        tensordict = tensordict.exclude("loc", "scale", "_feature", "context", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"][:, -1]
        with torch.no_grad():
            self.history_encoder(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        dones = (
            tensordict[("next", "done")]
            .expand(-1, -1, self.n_agents)
            .unsqueeze(-1)
        )
        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = compute_gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
        self.history_encoder(tensordict)
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        loss = policy_loss + entropy_loss + value_loss
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        self.encoder_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()
        self.encoder_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]