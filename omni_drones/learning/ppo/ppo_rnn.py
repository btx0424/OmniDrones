# MIT License
#
# Copyright (c) 2023 Botian Xu
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

from dataclasses import dataclass, replace
from functools import partial
from typing import Union

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec
from torchrl.envs import CatTensors, TensorDictPrimer
from torchrl.modules import ProbabilisticActor

from ..modules.distributions import IndependentNormal

from ..utils.gae import compute_gae
from ..utils.valuenorm import ValueNorm1


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


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, skip_conn) -> None:
        super().__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.skip_conn = skip_conn

    def forward(
        self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor
    ):
        T = x.shape[1]
        hx, cx = hx[:, 0], cx[:, 0]
        output = []
        for i in range(T):
            hx, cx = self._forward(x[:, i], is_init[:, i], hx, cx)
            output.append(hx)
        output = torch.stack(output, dim=1)
        output = self.ln(output)
        if self.skip_conn == "add":
            output = x + output
        elif self.skip_conn == "cat":
            output = torch.cat([x, output], dim=-1)
        return (
            output,
            hx.unsqueeze(1).expand(-1, T, *hx.shape[1:]),
            cx.unsqueeze(1).expand(-1, T, *cx.shape[1:]),
        )

    def _forward(
        self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor
    ):
        batch_shape = x.shape[:-1]
        reset = (
            1 - is_init.reshape(is_init.shape + (1,) * (hx.ndim - is_init.ndim)).float()
        )
        hx = hx * reset
        cx = cx * reset
        x = x.reshape(-1, self.lstm.input_size)
        hx = hx.reshape(-1, self.lstm.hidden_size)
        cx = cx.reshape(-1, self.lstm.hidden_size)
        hx, cx = self.lstm(x, (hx, cx))
        hx = hx.reshape(*batch_shape, -1)
        cx = cx.reshape(*batch_shape, -1)
        return hx, cx


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, skip_conn, bptt_len: int = 8) -> None:
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.skip_conn = skip_conn
        # self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor):
        T = x.shape[1]
        hx = hx[:, 0]
        output = []
        for i in range(T):
            hx = self._forward(x[:, i], is_init[:, i], hx)
            output.append(hx)
        output = torch.stack(output, dim=1)
        # output = self.ln(output)
        if self.skip_conn == "add":
            output = x + output
        elif self.skip_conn == "cat":
            output = torch.cat([x, output], dim=-1)
        return output, hx.unsqueeze(1).expand(-1, T, *hx.shape[1:])

    def _forward(self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor):
        batch_shape = x.shape[:-1]
        reset = (
            1 - is_init.reshape(is_init.shape + (1,) * (hx.ndim - is_init.ndim)).float()
        )
        hx = hx * reset
        x = x.reshape(-1, x.shape[-1])
        hx = hx.reshape(-1, hx.shape[-1])
        hx = self.gru(x, hx)
        hx = hx.reshape(*batch_shape, -1)
        return hx


@dataclass
class PPOConfig:
    name: str = "ppo_rnn"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    seq_len: int = 16

    # whether to take in priviledged infomation
    priv: bool = False

    rnn: str = "gru"
    skip_conn: Union[str, None] = None
    hidden_size: int = 128

    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo_gru", node=PPOConfig, group="algo")
cs.store("ppo_lstm", node=PPOConfig(rnn="lstm"), group="algo")


class PPORNNPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: CompositeSpec,
        action_spec: TensorSpec,
        reward_spec: TensorSpec,
        device,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = 0.001
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec.shape[-2:]

        fake_input = observation_spec.zero()

        if cfg.rnn == "gru":

            def make_rnn(branch: str):
                return TensorDictModule(
                    GRU(256, self.cfg.hidden_size, self.cfg.skip_conn),
                    ["feature", "is_init", f"{branch}_hx"],
                    ["feature", ("next", f"{branch}_hx")],
                )

        elif cfg.rnn == "lstm":

            def make_rnn(branch: str):
                return TensorDictModule(
                    LSTM(256, self.cfg.hidden_size, self.cfg.skip_conn),
                    ["feature", "is_init", f"{branch}_hx", f"{branch}_cx"],
                    ["feature", ("next", f"{branch}_hx"), ("next", f"{branch}_cx")],
                )

        else:
            raise NotImplementedError(self.cfg.rnn)

        if cfg.priv:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]

            def make_encoder():
                return TensorDictSequential(
                    TensorDictModule(
                        nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])), 
                        [("agents", "intrinsics")], ["context"]
                    ),
                    CatTensors(["feature", "context"], "feature", del_keys=False),
                )

            actor = TensorDictSequential(
                TensorDictModule(
                    make_mlp([256, 256]), [("agents", "observation")], ["feature"]
                ),
                make_rnn(branch="actor"),
                make_encoder(),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), Actor(self.action_dim)),
                    ["feature"], ["loc", "scale"]
                ),
            ).to(self.device)
            self.critic = TensorDictSequential(
                TensorDictModule(
                    make_mlp([256, 256]), [("agents", "observation")], ["feature"]
                ),
                make_rnn(branch="critic"),
                make_encoder(),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)),
                    ["feature"], ["state_value"]
                ),
            ).to(self.device)
        else:
            actor = TensorDictSequential(
                TensorDictModule(
                    make_mlp([256, 256]), [("agents", "observation")], ["feature"]
                ),
                make_rnn(branch="actor"),
                TensorDictModule(Actor(self.action_dim), ["feature"], ["loc", "scale"]),
            ).to(self.device)
            self.critic = TensorDictSequential(
                TensorDictModule(
                    make_mlp([256, 256]), [("agents", "observation")], ["feature"]
                ),
                make_rnn(branch="critic"),
                TensorDictModule(nn.LazyLinear(1), ["feature"], ["state_value"]),
            ).to(self.device)

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

        self._maybe_init_state(fake_input)
        self.actor(fake_input.unsqueeze(1))
        self.critic(fake_input.unsqueeze(1))

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.0)
                elif isinstance(module, (nn.GRUCell, nn.LSTMCell)):
                    nn.init.orthogonal_(module.weight_hh)

            self.actor.apply(init_)
            self.critic.apply(init_)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        self.value_norm = ValueNorm1(reward_spec.shape[-2:]).to(self.device)

    def _maybe_init_state(self, tensordict: TensorDict):
        shape = tensordict.get(("agents", "observation")).shape[:-1]
        if self.cfg.rnn == "gru":
            for key in ("actor_hx", "critic_hx"):
                if key not in tensordict.keys():
                    tensordict.set(
                        key,
                        torch.zeros(*shape, self.cfg.hidden_size, device=self.device),
                    )
        elif self.cfg.rnn == "lstm":
            for key in ("actor_hx", "actor_cx", "critic_hx", "critic_cx"):
                if key not in tensordict.keys():
                    tensordict.set(
                        key,
                        torch.zeros(*shape, self.cfg.hidden_size, device=self.device),
                    )
        return tensordict

    def __call__(self, tensordict: TensorDict):
        self._maybe_init_state(tensordict)
        tensordict = tensordict.unsqueeze(1)  # dummy time dimension
        tensordict = self.actor(tensordict)
        tensordict = self.critic(tensordict)
        tensordict = tensordict.squeeze(1)
        tensordict = tensordict.exclude("loc", "scale", "feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"][:, [-1]]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"].squeeze(1)
        rewards = tensordict[("next", "agents", "reward")]
        dones = (
            tensordict[("next", "terminated")]
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
            batch = make_batch(tensordict, self.cfg.num_minibatches, self.cfg.seq_len)
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
        surr2 = adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        policy_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = -self.entropy_coef * torch.mean(entropy)

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
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), 5
        )
        self.actor_opt.step()
        self.critic_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict(
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "actor_grad_norm": actor_grad_norm,
                "critic_grad_norm": critic_grad_norm,
                "explained_var": explained_var,
            },
            [],
        )


def make_batch(tensordict: TensorDict, num_minibatches: int, seq_len: int = -1):
    if seq_len > 1:
        N, T = tensordict.shape
        T = (T // seq_len) * seq_len
        tensordict = tensordict[:, :T].reshape(-1, seq_len)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
    else:
        tensordict = tensordict.reshape(-1)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
