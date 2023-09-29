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

class TD3Policy(object):

    def __init__(self,
        cfg,
        agent_spec: AgentSpec,
        device: str="cuda",
    ) -> None:
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        self.gradient_steps = int(cfg.gradient_steps)
        self.batch_size = int(cfg.batch_size)
        self.buffer_size = int(cfg.buffer_size)
        
        self.target_noise = self.cfg.target_noise
        self.policy_noise = self.cfg.policy_noise
        self.noise_clip = self.cfg.noise_clip

        self.obs_name = ("agents", "observation")
        self.act_name = ("agents", "action")
        self.reward_name = ("agents", "reward")

        self.action_dim = self.agent_spec.action_spec.shape[-1]
        self.make_model()        

        self.replay_buffer = TensorDictReplayBuffer(
            batch_size=self.batch_size,
            storage=LazyTensorStorage(max_size=self.buffer_size, device=self.device),
            sampler=RandomSampler(),
        )
    
    def make_model(self):

        self.policy_in_keys = [self.obs_name]
        self.policy_out_keys = [self.act_name, f"{self.agent_spec.name}.logp"]
        
        encoder = make_encoder(self.cfg.actor, self.agent_spec.observation_spec)
        self.actor = TensorDictModule(
            nn.Sequential(
                encoder,
                nn.LazyLinear(self.action_dim),
                nn.Tanh()
            ),
            in_keys=self.policy_in_keys, out_keys=self.policy_out_keys
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor.lr)

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
        self.critic_loss_fn = {"mse":F.mse_loss, "smooth_l1": F.smooth_l1_loss}[self.cfg.critic_loss]

    def __call__(self, tensordict: TensorDict, deterministic: bool=False) -> TensorDict:
        actor_input = tensordict.select(*self.policy_in_keys)
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]
        actor_output = self.actor(actor_input)
        action_noise = (
            actor_output[self.act_name]
            .clone()
            .normal_(0, self.policy_noise)
            .clamp_(-self.noise_clip, self.noise_clip)
        )
        actor_output[self.act_name].add_(action_noise)
        tensordict.update(actor_output)
        return tensordict

    def train_op(self, data: TensorDict, verbose: bool=False):
        self.replay_buffer.extend(data.reshape(-1))

        if len(self.replay_buffer) < self.cfg.buffer_size:
            print(f"filling buffer: {len(self.replay_buffer)}/{self.cfg.buffer_size}")
            return {}
        
        infos_critic = []
        infos_actor = []

        t = range(1, self.gradient_steps+1)

        for gradient_step in tqdm(t) if verbose else t:

            transition = self.replay_buffer.sample(self.batch_size)

            state   = transition[("agents", "observation")]
            action_taken = transition[("agents", "action")]

            reward  = transition[("next", "agents", "reward")]
            next_dones  = transition[("next", "done")].float().unsqueeze(-1)
            next_state  = transition[("next", "agents", "observation")]

            with torch.no_grad():

                next_action: torch.Tensor = self.actor_target(transition["next"])[self.act_name]
                action_noise = (
                    next_action
                    .clone()
                    .normal_(0, self.target_noise)
                    .clamp_(-self.noise_clip, self.noise_clip)
                )
                next_action = torch.clamp(next_action + action_noise, -1, 1)

                next_qs = self.critic_target(next_state, next_action)
                next_q = torch.min(next_qs, dim=-1, keepdim=True).values
                target_q = (reward + self.cfg.gamma * (1 - next_dones) * next_q).detach().squeeze(-1)
                assert not torch.isinf(target_q).any()
                assert not torch.isnan(target_q).any()

            qs = self.critic(state, action_taken)
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
                    actor_output = self.actor(transition)
                    action = actor_output[self.act_name]

                    qs = self.critic(state, action)
                    q = torch.min(qs, dim=-1, keepdim=True).values
                    actor_loss = - q.mean()

                    self.actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                    self.actor_opt.step()

                    infos_actor.append(TensorDict({
                        "actor_loss": actor_loss,
                        "actor_grad_norm": actor_grad_norm,
                    }, []))
                
                with torch.no_grad():
                    soft_update(self.actor_target, self.actor, self.cfg.tau)
                    soft_update(self.critic_target, self.critic, self.cfg.tau)
        
        infos = {**torch.stack(infos_actor), **torch.stack(infos_critic)}
        infos = {k: torch.mean(v).item() for k, v in infos.items()}
        return infos
    
    def state_dict(self):
        state_dict = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict()
        }
        return state_dict

from .modules.networks import MLP
from .modules.distributions import TanhIndependentNormalModule
from .common import make_encoder


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

