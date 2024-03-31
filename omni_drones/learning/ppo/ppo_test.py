# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union
import einops

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import GAE

@dataclass
class PPOConfig:
    name: str = "ppo_test"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16

    max_over: str = "value" # (value, advantage)
    adv_key: str = "adv"

    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo_test", node=PPOConfig, group="algo")

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

REWARD_KEY = ("next", "reward")
DONE_KEY = ("next", "terminated")

class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale


class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = 0.001
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec.shape[-2:]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()
        
        actor_module = TensorDictModule(
            nn.Sequential(make_mlp([256, 256, 256]), Actor(self.action_dim)),
            [("agents", "policy")], ["loc", "scale"]
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic = TensorDictModule(
            nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1)),
            [("agents", "critic")], ["state_value"]
        ).to(self.device)

        self.critic_priv = TensorDictModule(
            nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1)),
            [("agents", "policy")], ["state_value"]
        ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)
        self.critic_priv(fake_input)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
            
            self.actor.apply(init_)
            self.critic.apply(init_)
            self.critic_priv.apply(init_)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.critic_priv.parameters()),
            lr=5e-4
        )
        self.value_norm = ValueNorm1(reward_spec["reward"].shape[-1]).to(self.device)
    
    def __call__(self, tensordict: TensorDict):
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        self._compute_advantage(tensordict)

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        infos = {
            k: torch.mean(v.float()).item()
            for k, v in sorted(torch.stack(infos).items())
        }
        infos["priv_larger"] = tensordict["priv_larger"].float().mean().item()
        return infos

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict[self.cfg.adv_key]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(tensordict["ret"], values)

        values_priv = self.critic_priv(tensordict)["state_value"]
        value_loss_priv = self.critic_loss_fn(tensordict["ret_priv"], values_priv)

        loss = policy_loss + entropy_loss + value_loss + value_loss_priv
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()
        explained_var = 1 - F.mse_loss(values, tensordict["ret"]) / tensordict["ret"].var()
        explained_var_priv = 1 - F.mse_loss(values_priv, tensordict["ret_priv"]) / tensordict["ret_priv"].var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "value_loss_priv": value_loss_priv,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var,
            "explained_var_priv": explained_var_priv,
        }, [])

    @torch.no_grad()
    def _compute_advantage(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        
        values = self.critic(tensordict)["state_value"]
        values_priv = self.critic_priv(tensordict)["state_value"]

        next_values = self.critic(next_tensordict)["state_value"]
        next_values_priv = self.critic_priv(next_tensordict)["state_value"]

        values = self.value_norm.denormalize(values)
        values_priv = self.value_norm.denormalize(values_priv)
        next_values = self.value_norm.denormalize(next_values)
        next_values_priv = self.value_norm.denormalize(next_values_priv)

        rewards = tensordict[REWARD_KEY]
        dones = tensordict[DONE_KEY]

        values_max = torch.max(values, values_priv)
        next_values_max = torch.max(next_values, next_values_priv)
        
        adv, ret = self.gae(rewards, dones, values_max, next_values_max)
        adv_priv, ret_priv = self.gae(rewards, dones, values_priv, next_values_priv)

        if self.cfg.max_over == "value":
            priv_larger = values_priv > values
            adv_max, ret_max = self.gae(rewards, dones, values_max, next_values_max)
        else:
            priv_larger = adv_priv > adv
            adv_max = torch.max(adv, adv_priv)
            ret_max = torch.max(ret, ret_priv)
        
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)
        ret_priv = self.value_norm.normalize(ret_priv)
        ret_max = self.value_norm.normalize(ret_max)

        adv = normalize(adv)
        adv_priv = normalize(adv_priv)
        adv_max = normalize(adv_max)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)
        tensordict.set("adv_priv", adv_priv)
        tensordict.set("ret_priv", ret_priv)
        tensordict.set("adv_max", adv_max)
        tensordict.set("ret_max", ret_max)
        tensordict.set("priv_larger", priv_larger)
        return tensordict


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]

def normalize(x: torch.Tensor):
    return (x - x.mean()) / x.std().clamp(1e-7)