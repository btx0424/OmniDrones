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

"""
This is a more concise and readable implementation of MAPPO using TorchRL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from tensordict import TensorDict
from tensordict.nn import (
    EnsembleModule as _EnsembleModule, 
    TensorDictSequential,
    TensorDictModule, 
    TensorDictModuleBase, 
    make_functional, 
    TensorDictParams
)
from torchrl.modules import ProbabilisticActor
from torchrl.data import TensorSpec, CompositeSpec
from torchrl.envs.transforms import CatTensors
from einops.layers.torch import Rearrange, Reduce

from .ppo.common import GAE, make_mlp
from .modules.distributions import IndependentNormal
from .utils.valuenorm import ValueNorm1

def make_transformer(
    obs_spec: CompositeSpec, 
    embed_dim: int=128,
    nhead: int=1,
    num_layers: int=1
):
    
    embedding_keys = []
    embeddings = []

    for key, spec in obs_spec.items(True, True):
        embeddings.append(TensorDictModule(nn.LazyLinear(embed_dim), [key], [key + "_embedding"]))
        embedding_keys.append(key + "_embedding")

    encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=256)
    encoder = nn.Sequential(
        nn.TransformerEncoder(encoder_layer, num_layers=num_layers),
        Reduce("b t n e -> b n e", "mean")
    )

    module = TensorDictSequential(
        *embeddings,
        CatTensors(embedding_keys, "embedding", del_keys=True),
        TensorDictModule(encoder, ["embedding"], ["embedding"]),
    )

    return module

    
class Actor(nn.Module):
    def __init__(self, action_dim: int, predict_std: bool=False) -> None:
        super().__init__()
        self.predict_std = predict_std
        if predict_std:
            self.actor_mean = nn.LazyLinear(action_dim * 2)
        else:
            self.actor_mean = nn.LazyLinear(action_dim)
            self.actor_std = nn.Parameter(torch.zeros(action_dim))
        self.scale_mapping = torch.exp
    
    def forward(self, features: torch.Tensor):
        if self.predict_std:
            loc, scale = self.actor_mean(features).chunk(2, dim=-1)
        else:
            loc = self.actor_mean(features)
            scale = self.actor_std.expand_as(loc)
        scale = self.scale_mapping(scale)
        return loc, scale


class EnsembleModule(_EnsembleModule):

    def __init__(self, module: TensorDictModuleBase, num_copies: int):
        super(_EnsembleModule, self).__init__()
        self.in_keys = module.in_keys
        self.out_keys = module.out_keys
        self.num_copies = num_copies

        params_td = make_functional(module).expand(num_copies).to_tensordict()
        self.module = module
        self.vmapped_forward = torch.vmap(self.module, (1, 0), 1)
        self.reset_parameters_recursive(params_td)
        self.params_td = TensorDictParams(params_td)
    
    def forward(self, tensordict: TensorDict):
        tensordict = tensordict.select(*self.in_keys)
        tensordict.batch_size = [tensordict.shape[0], self.num_copies]
        return self.vmapped_forward(tensordict, self.params_td)


def init_(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, 0.01)
        nn.init.constant_(module.bias, 0.)


class MAPPO:

    def __init__(
        self, 
        cfg, 
        observation_spec: CompositeSpec, 
        action_spec: TensorSpec, 
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.entropy_coef = 0.001
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.gae = GAE(0.99, 0.95)
        
        if not action_spec.ndim > 2:
            raise ValueError("Please use PPOPolicy for single-agent environments.")
        
        self.num_agents, self.action_dim = action_spec.shape[-2:]
        fake_input = observation_spec.zero()

        actor_module = TensorDictModule(
            nn.Sequential(
                make_mlp([256, 256], nn.Mish),
                Actor(self.action_dim)
            ),
            [("agents", "observation")], ["loc", "scale"]
        ).to(self.device)
        actor_module(fake_input)
        
        if not cfg.share_actor:
            actor_module = EnsembleModule(actor_module, self.num_agents)
        else:
            actor_module.apply(init_)

        self.actor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic = TensorDictModule(
            nn.Sequential(
                make_mlp([512, 256], nn.Mish),
                nn.LazyLinear(self.num_agents),
                Rearrange("... -> ... 1")
            ),
            [("agents", "observation_central")], ["state_value"]
        ).to(self.device)
        self.critic(fake_input)
        self.critic.apply(init_)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        self.value_norm = ValueNorm1(input_shape=1).to(self.device)
    
    def __call__(self, tensordict: TensorDict):
        tensordict.update(self.actor(tensordict))
        self.critic(tensordict)
        return tensordict
    
    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        dones = tensordict[("next", "terminated")]
        dones = einops.repeat(dones, "t n 1 -> t n a 1", a=self.num_agents)
        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
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
        tensordict = tensordict[~tensordict["is_init"].squeeze(1)]
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * entropy

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
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()
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