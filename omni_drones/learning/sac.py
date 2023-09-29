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
from tensordict.nn import TensorDictModule
import functorch
import numpy as np

from torchrl.data import (
    TensorSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec as UnboundedTensorSpec,
    CompositeSpec,
    TensorDictReplayBuffer
)
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.objectives.utils import hold_out_net

import copy
from tqdm import tqdm
from omni_drones.utils.torchrl import AgentSpec
from tensordict import TensorDict
from .common import soft_update

class SACPolicy(object):

    def __init__(self,
        cfg,
        agent_spec: AgentSpec,
        device: str="cuda",
    ) -> None:
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        self.gradient_steps = int(cfg.gradient_steps)
        self.buffer_size = int(cfg.buffer_size)
        self.batch_size = int(cfg.batch_size)

        self.obs_name = ("agents", "observation")
        self.act_name = ("agents", "action")
        self.reward_name = ("agents", "reward")

        self.make_actor()
        self.make_critic()
        
        self.action_dim = self.agent_spec.action_spec.shape[-1]
        self.target_entropy = - torch.tensor(self.action_dim, device=self.device)
        init_entropy = 1.0
        self.log_alpha = nn.Parameter(torch.tensor(init_entropy, device=self.device).log())
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)

        self.replay_buffer = TensorDictReplayBuffer(
            batch_size=self.batch_size,
            storage=LazyTensorStorage(max_size=self.buffer_size, device=self.device),
            sampler=RandomSampler(),
        )
    
    def make_actor(self):

        self.policy_in_keys = [self.obs_name]
        self.policy_out_keys = [self.act_name, f"{self.agent_spec.name}.logp"]
        
        if self.cfg.share_actor:
            self.actor = TensorDictModule(
                Actor(
                    self.cfg.actor, 
                    self.agent_spec.observation_spec, 
                    self.agent_spec.action_spec
                ),
                in_keys=self.policy_in_keys, out_keys=self.policy_out_keys
            ).to(self.device)
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor.lr)
        else:
            raise NotImplementedError


    def make_critic(self):
        self.value_in_keys = [self.obs_name, self.act_name]
        self.value_out_keys = [f"{self.agent_spec.name}.q"]

        self.critic = Critic(
            self.cfg.critic, 
            1,
            self.agent_spec.observation_spec,
            self.agent_spec.action_spec
        ).to(self.device)
        
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic.lr)       
        self.critic_loss_fn = {"mse": F.mse_loss, "smooth_l1": F.smooth_l1_loss}[self.cfg.critic_loss]

    def __call__(self, tensordict: TensorDict, deterministic: bool=False) -> TensorDict:
        # return tensordict.update({self.act_name: self.agent_spec.action_spec.zero()})
        actor_input = tensordict.select(*self.policy_in_keys)
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]
        actor_output = self.actor(actor_input)
        tensordict.update(actor_output)
        return tensordict

    def train_op(self, data: TensorDict, verbose: bool=False):
        self.replay_buffer.extend(data.reshape(-1))

        if len(self.replay_buffer) < self.cfg.buffer_size:
            print(f"filling buffer: {len(self.replay_buffer)} < {self.cfg.buffer_size}")
            return {}
        
        infos_critic = []
        infos_actor = []

        t = range(1, self.gradient_steps+1)
       
        for gradient_step in tqdm(t) if verbose else t:

            transition = self.replay_buffer.sample()

            state   = transition[self.obs_name]
            actions = transition[self.act_name]

            reward  = transition[("next", "agents", "reward")]
            next_dones  = transition[("next", "done")].float().unsqueeze(-1)
            next_state  = transition[("next", "agents", "observation")]

            with torch.no_grad():
                actor_output = self.actor(transition["next"], deterministic=False)
                next_act = actor_output[self.act_name]
                next_logp = actor_output[f"{self.agent_spec.name}.logp"]
                next_qs = self.critic_target(next_state, next_act)
                next_q = torch.min(next_qs, dim=-1, keepdim=True).values
                next_q = next_q - self.log_alpha.exp() * next_logp
                target_q = (reward + self.cfg.gamma * (1 - next_dones) * next_q).detach().squeeze(-1)
                assert not torch.isinf(target_q).any()
                assert not torch.isnan(target_q).any()

            qs = self.critic(state, actions)
            critic_loss = sum(self.critic_loss_fn(q, target_q) for q in qs.unbind(-1))
            self.critic_opt.zero_grad()
            critic_loss.backward()
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
            self.critic_opt.step()
            infos_critic.append(TensorDict({
                "critic_loss": critic_loss,
                "critic_grad_norm": critic_grad_norm,
                "q_taken": qs.mean()
            }, []))

            if (gradient_step + 1) % self.cfg.actor_delay == 0:

                with hold_out_net(self.critic):
                    actor_output = self.actor(transition, deterministic=False)
                    act = actor_output[self.act_name]
                    logp = actor_output[f"{self.agent_spec.name}.logp"]

                    qs = self.critic(state, act)
                    q = torch.min(qs, dim=-1).values
                    actor_loss = (self.log_alpha.exp() * logp - q).mean()
                    self.actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                    self.actor_opt.step()

                    self.alpha_opt.zero_grad()
                    alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
                    alpha_loss.backward()
                    self.alpha_opt.step()

                    infos_actor.append(TensorDict({
                        "actor_loss": actor_loss,
                        "actor_grad_norm": actor_grad_norm,
                        "entropy": -logp.mean(),
                        "alpha": self.log_alpha.exp().detach(),
                        "alpha_loss": alpha_loss,
                    }, []))


            if (gradient_step + 1) % self.cfg.target_update_interval == 0:
                with torch.no_grad():
                    soft_update(self.critic_target, self.critic, self.cfg.tau)
        
        infos = {**torch.stack(infos_actor), **torch.stack(infos_critic)}
        infos = {k: torch.mean(v).item() for k, v in infos.items()}
        return infos

    def state_dict(self):
        state_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "ctitic_target": self.critic_target.state_dict()
        }
        return state_dict
    
from .modules.networks import MLP
from .modules.distributions import TanhIndependentNormalModule
from .common import make_encoder

class Actor(nn.Module):
    def __init__(self, 
        cfg,
        observation_spec: TensorSpec, 
        action_spec: BoundedTensorSpec,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = make_encoder(cfg, observation_spec)
        
        self.act = TanhIndependentNormalModule(
            self.encoder.output_shape.numel(), 
            action_spec.shape[-1], 
        )

    def forward(self, obs: torch.Tensor, deterministic: bool=False):
        x = self.encoder(obs)
        act_dist = self.act(x)

        if deterministic:
            act = act_dist.mode
        else:
            act = act_dist.rsample()
        log_prob = act_dist.log_prob(act).unsqueeze(-1)

        return act, log_prob


class Critic(nn.Module):
    def __init__(self, 
        cfg,
        num_agents: int,
        state_spec: TensorSpec,
        action_spec: BoundedTensorSpec, 
        num_critics: int = 2,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_agents = num_agents
        self.act_space = action_spec
        self.state_space = state_spec
        self.num_critics = num_critics

        self.critics = nn.ModuleList([
            self._make_critic() for _ in range(self.num_critics)
        ])

    def _make_critic(self):
        if isinstance(self.state_space, (BoundedTensorSpec, UnboundedTensorSpec)):
            action_dim = self.act_space.shape[-1]
            state_dim = self.state_space.shape[-1]
            num_units = [
                action_dim * self.num_agents + state_dim, 
                *self.cfg["hidden_units"]
            ]
            base = MLP(num_units)
        else:
            raise NotImplementedError
        
        v_out = nn.Linear(base.output_shape.numel(), 1)
        return nn.Sequential(base, v_out)
        
    def forward(self, state: torch.Tensor, actions: torch.Tensor):
        """
        Args:
            state: (batch_size, state_dim)
            actions: (batch_size, num_agents, action_dim)
        """
        state = state.flatten(1)
        actions = actions.flatten(1)
        x = torch.cat([state, actions], dim=-1)
        return torch.stack([critic(x) for critic in self.critics], dim=-1)


