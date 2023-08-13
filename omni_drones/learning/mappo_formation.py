import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import CompositeSpec
from torchrl.envs.transforms import CatTensors, ExcludeTransform
from torchrl.modules import ProbabilisticActor
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from .utils.gae import compute_gae
from .utils.valuenorm import ValueNorm1
from .modules.distributions import IndependentNormal

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.ELU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            make_mlp([256, 128]), 
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.LazyLinear(128)
        )
        self.actor_mean = nn.Sequential(
            nn.LazyLinear(128),
            nn.ELU(),
            nn.LazyLinear(action_dim)
        )
        self.actor_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs_state: torch.Tensor, obs_map: torch.Tensor):
        feature_state = self.mlp(obs_state)
        feature_map = self.cnn(obs_map)
        feature = torch.cat([feature_state, feature_map], dim=-1)
        loc = self.actor_mean(feature)
        scale = torch.exp(self.actor_std)
        return loc, torch.ones_like(loc) * scale


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            make_mlp([256, 128]), 
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.LazyLinear(128)
        )
        self.value = nn.Sequential(
            nn.LazyLinear(128),
            nn.ELU(),
            nn.LazyLinear(1)
        )
    
    def forward(self, obs_state: torch.Tensor, obs_map: torch.Tensor):
        feature_state = self.mlp(obs_state)
        feature_map = self.cnn(obs_map)
        feature = torch.cat([feature_state, feature_map], dim=-1)
        value = self.value(feature)
        return value


class Vmap(nn.Module):
    def __init__(self, module: nn.Module, in_dims, out_dims, randomness="different"):
        super().__init__()
        from functorch import vmap
        self.module = module
        self.forward_fn = vmap(module, in_dims, out_dims, randomness)

    def forward(self, *args, **kwargs):
        return self.forward_fn(*args, **kwargs)


class PPOFormation:
    def __init__(self, cfg, agent_spec, device):
        self.device = device
        self.agent_spec = agent_spec

        self.entropy_coef = 0.001
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.action_dim = self.agent_spec.action_spec.shape[-1]

        actor = Actor(self.agent_spec.action_spec.shape[-1]).to(self.device)
        critic = Critic().to(self.device)
        fake_input = agent_spec.observation_spec[0].zero()
        actor(fake_input["obs_self"], fake_input["obs_minimap"])
        critic(fake_input["obs_self"], fake_input["obs_minimap"])

        def init_(module):
            if isinstance(module, nn.Linear):
                # nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictModule(
                Vmap(actor, in_dims=1, out_dims=1),
                in_keys=[
                    ("agents", "observation", "obs_self"),
                    ("agents", "observation", "obs_minimap")
                ],
                out_keys=["loc", "scale"]
            ),
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        self.critic = TensorDictModule(
            module=Vmap(critic, in_dims=1, out_dims=1),
            in_keys=[
                ("agents", "observation", "obs_self"),
                ("agents", "observation", "obs_minimap")
            ],
            out_keys=["state_value"]
        ).to(self.device)
        parameters = [
            *self.actor.parameters(),
            *self.critic.parameters(),
        ]
        self.opt = torch.optim.Adam(parameters, lr=5e-4)
        self.value_norm = ValueNorm1(self.agent_spec.reward_spec.shape[-2:]).to(self.device)
    
    def __call__(self, tensordict: TensorDict):
        tensordict = self.actor(tensordict)
        tensordict = self.critic(tensordict)
        tensordict = tensordict.exclude("loc", "scale")
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"][:, -1]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        dones = (
            tensordict[("next", "done")]
            .expand(-1, -1, self.agent_spec.n)
            .unsqueeze(-1)
        ).clone()
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
        for epoch in range(4):
            batch = make_batch(tensordict, 16)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
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
        self.opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.opt.param_groups[0]["params"], 10.
        )
        self.opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "grad_norm": grad_norm,
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