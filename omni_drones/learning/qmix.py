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
from tensordict import TensorDict
from tensordict.utils import expand_right
from tensordict.nn import TensorDictModule, TensorDictSequential
from functorch import vmap
from torchrl.data import (
    DiscreteTensorSpec,
)
from .modules.rnn import GRU
from .modules.networks import ENCODERS_MAP, MLP
from .common import MyBuffer, soft_update, make_encoder
import copy
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict

class QMIXPolicy:
    def __init__(
        self,
        cfg,
        agent_spec,
        device=None,
    ):
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device
        self.agent_name = agent_spec.name

        n_agents = agent_spec.n
        if not isinstance(agent_spec.action_spec, DiscreteTensorSpec):
            raise ValueError("Only discrete action spaces are supported for QMIX.")
        
        num_actions = agent_spec.action_spec.space.n
        obs_name = f"{self.agent_name}.obs"
        self.action_name = ("action", f"{self.agent_name}.action")
        self.state_name = (
            f"{self.agent_name}.state" 
            if self.agent_spec.state_spec is not None 
            else f"{self.agent_name}.obs"
        )

        obs_encoder = make_encoder(cfg.q_net, agent_spec.observation_spec)
        hidden_dim = cfg.q_net.hidden_dim
        self.agent_q = TensorDictSequential(
            TensorDictModule(obs_encoder, [obs_name], ["hidden"]),
            TensorDictModule(
                GRU(obs_encoder.output_shape.numel(), hidden_dim),
                ["hidden", f"{self.agent_name}.rnn_state", "is_init"],
                ["hidden", f"{self.agent_name}.rnn_state"],
            ),
            TensorDictModule(
                nn.Linear(hidden_dim, num_actions), ["hidden"], [f"{self.agent_name}.q"]
            ),
        ).to(self.device)
        self.target_agent_q = copy.deepcopy(self.agent_q)

        self.action_selector = TensorDictModule(
            EpsilonGreedyActionSelector(anneal_time=self.cfg.anneal_time), 
            [f"{self.agent_name}.q", "epsilon_t"], 
            [self.action_name, "epsilon"]
        )

        if agent_spec.state_spec is not None:
            state_encoder = make_encoder(cfg.q_mixer, agent_spec.state_spec)
        else:
            state_encoder = make_encoder(cfg.q_mixer, agent_spec.observation_spec)
        hidden_dim = cfg.q_mixer.hidden_dim
        mixer = QMIXer(n_agents, state_encoder.output_shape.numel(), hidden_dim)
        self.mixer = TensorDictSequential(
            TensorDictModule(state_encoder, [self.state_name], ["mixer_hidden"]),
            TensorDictModule(mixer, [f"chosen_q", "mixer_hidden"], [f"q_tot"]),
        ).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer)

        params = list(self.agent_q.parameters()) + list(self.mixer.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg.lr)

        self.rb = MyBuffer(cfg.buffer_size, device="cpu") # device=self.device)
        
        self.t = 0 # for epsilon annealing

    def __call__(self, tensordict: TensorDict):
        tensordict.set(
            "epsilon_t", 
            torch.full([*tensordict.shape, self.agent_spec.n], self.t, device=self.device)
        )
        tensordict.update(self._call(tensordict, self.agent_q, 1))
        self.action_selector(tensordict)

        if tensordict.get("_reset", None) is not None:
            _reset = tensordict["_reset"]
            self.t += _reset.sum().item()
        return tensordict

    def _call(self, tensordict: TensorDict, agent_q: TensorDictSequential, vmap_dim: int):
        q_input = tensordict.select(*agent_q.in_keys, strict=False) 
        q_input["is_init"] = expand_right(
            q_input["is_init"], (*q_input.batch_size, self.agent_spec.n)
        )
        q_input.batch_size = [*q_input.batch_size, self.agent_spec.n]
        q_output: TensorDict = vmap(agent_q, in_dims=vmap_dim, out_dims=vmap_dim, randomness="different")(q_input)
        return q_output

    def train_op(self, tensordict: TensorDict):
        reward = tensordict[("next", "reward", f"{self.agent_name}.reward")]
        if reward.dim() == 4: # [N, L, M, *]
            # force shared reward
            tensordict = tensordict.clone().set(
                ("next", "reward", f"{self.agent_name}.reward"),
                reward.sum(dim=(-1, -2)).unsqueeze(-1)
            )

        self.rb.extend(tensordict.cpu())

        if len(self.rb) < self.cfg.buffer_size:
            print(f"{len(self.rb)} < {self.cfg.buffer_size}")
            return {}

        infos = defaultdict(list)
        t = tqdm(range(self.cfg.gradient_steps))
        for gradient_step in t:
            batch: TensorDict = self.rb.sample(self.cfg.batch_size).to(self.device)
            chosen_actions = batch[self.action_name]  # [N, L, M, 1]
            reward = batch[("next", "reward", f"{self.agent_name}.reward")]
            next_done = batch[("next", "done")].float()

            qs = self._call(batch, self.agent_q, 2)[f"{self.agent_name}.q"]  # [N, L, M, |A|]
            chosen_action_qs = torch.gather(qs, -1, chosen_actions)

            with torch.no_grad():
                target_qs: torch.Tensor = self._call(batch["next"], self.target_agent_q, 2)[f"{self.agent_name}.q"]
                target_max_qs = target_qs.max(dim=-1, keepdim=True).values

            chosen_action_q_tot = self.mixer(
                batch.set("chosen_q", chosen_action_qs),
            )["q_tot"]
            target_action_q_tot = self.target_mixer(
                batch["next"].set("chosen_q", target_max_qs),
            )["q_tot"]

            loss = F.mse_loss(
                chosen_action_q_tot,
                reward + self.cfg.gamma * target_action_q_tot * (1 - next_done),
            )
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            infos["q_loss"].append(loss)
            infos["q_taken_mean"].append(chosen_action_qs.mean())
            infos["q_tot_taken_mean"].append(chosen_action_q_tot.mean())
            infos["epsilon"].append(batch["epsilon"].mean())
            t.set_postfix({"q_loss": loss.item()})

            if gradient_step % self.cfg.target_update_interval == 0:
                soft_update(self.agent_q, self.target_agent_q, self.cfg.tau)
                soft_update(self.mixer, self.target_mixer, self.cfg.tau)

        t.close()
        infos = {k: torch.stack(v).mean().item() for k, v in infos.items()}
        return infos

class EpsilonGreedyActionSelector(nn.Module):
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_finish: float = 0.05,
        anneal_time: float = 5000,
        decay="linear",
    ):
        super().__init__()
        assert decay in ["linear", "exp"]
        self.epsilon_start = epsilon_start
        self.epsilon_finish = epsilon_finish
        self.anneal_time = anneal_time
        self.decay = decay
        self.delta = (epsilon_start - epsilon_finish) / anneal_time

    def forward(self, agent_qs: torch.Tensor, t: torch.Tensor):
        if self.decay == "linear":
            epsilon = torch.clamp(self.epsilon_start - self.delta * t, self.epsilon_finish)
        elif self.decay == "exp":
            raise NotImplementedError

        num_actions = agent_qs.shape[-1]
        random_actions = torch.randint(
            0, num_actions, agent_qs.shape[:-1], device=agent_qs.device
        ).unsqueeze(-1)
        greedy_actions = agent_qs.argmax(dim=-1, keepdim=True)
        epsilon = epsilon.reshape_as(random_actions)
        assert epsilon.shape == random_actions.shape == greedy_actions.shape
        actions = torch.where(
            torch.rand_like(epsilon) < epsilon,
            random_actions,
            greedy_actions,
        )
        return actions, epsilon


class QMIXer(nn.Module):
    def __init__(
        self,
        n_agents: int,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.hyper_w1 = nn.Linear(self.input_dim, self.hidden_dim * self.n_agents)
        self.hyper_b1 = nn.Linear(self.input_dim, self.hidden_dim)
        
        self.hyper_w2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor):
        """
        
        agent_qs: [N, L, M, |A|]
        state: [N, L, state_dim]

        """
        assert agent_qs.shape[-2] == self.n_agents
        batch_shape = agent_qs.shape[:-2]
        
        agent_qs = agent_qs.reshape(batch_shape.numel(), 1, -1)
        state = state.reshape(batch_shape.numel(), -1)

        w1 = torch.abs(self.hyper_w1(state)).unflatten(-1, (self.n_agents, self.hidden_dim))
        b1 = self.hyper_b1(state).unsqueeze(-2)
        w2 = torch.abs(self.hyper_w2(state)).unsqueeze(-1)
        b2 = self.hyper_b2(state).unsqueeze(-2)

        h = F.elu(torch.bmm(agent_qs, w1) + b1)
        q_tot = torch.bmm(h, w2) + b2
        q_tot = q_tot.reshape(*batch_shape, 1)
        return q_tot
