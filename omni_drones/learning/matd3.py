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
from tensordict.nn import TensorDictModule, make_functional
from tensordict import TensorDict
from functorch import vmap

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
from .common import soft_update

class MATD3Policy(object):

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

        self.obs_name = f"{self.agent_spec.name}.obs"
        self.act_name = ("action", f"{self.agent_spec.name}.action")
        if agent_spec.state_spec is not None:
            self.state_name = f"{self.agent_spec.name}.state"
        else:
            self.state_name = f"{self.agent_spec.name}.obs"
        self.reward_name = f"{self.agent_spec.name}.reward"

        self.action_dim = self.agent_spec.action_spec.shape[-1]
        self.make_model()        

        self.replay_buffer = TensorDictReplayBuffer(
            batch_size=self.batch_size,
            storage=LazyTensorStorage(max_size=self.buffer_size, device="cpu"),
            sampler=RandomSampler(),
        )
    
    def make_model(self):

        self.policy_in_keys = [self.obs_name]
        self.policy_out_keys = [self.act_name, f"{self.agent_spec.name}.logp"]
        
        def create_actor():
            encoder = make_encoder(self.cfg.actor, self.agent_spec.observation_spec)
            return TensorDictModule(
                nn.Sequential(
                    encoder,
                    nn.ELU(),
                    nn.Linear(encoder.output_shape.numel(), self.action_dim),
                    nn.Tanh()
                ),
                in_keys=self.policy_in_keys, 
                out_keys=self.policy_out_keys
            ).to(self.device)
        
        if self.cfg.share_actor:
            self.actor = create_actor()
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor.lr)
            self.actor_params = make_functional(self.actor).expand(self.agent_spec.n)
            self.actor_target_params = self.actor_params.clone()
        else:
            actors = nn.ModuleList([create_actor() for _ in range(self.agent_spec.n)])
            self.actor = actors[0]
            self.actor_opt = torch.optim.Adam(actors.parameters(), lr=self.cfg.actor.lr)
            self.actor_params = torch.stack([make_functional(actor) for actor in actors])
            self.actor_target_params = self.actor_params.clone()

        if self.agent_spec.state_spec is not None:
            self.value_in_keys = [self.state_name, self.act_name]
            self.value_out_keys = [f"{self.agent_spec.name}.q"]

            self.critic = Critic(
                self.cfg.critic,
                self.agent_spec.n,
                self.agent_spec.state_spec,
                self.agent_spec.action_spec
            ).to(self.device)
        else:
            self.value_in_keys = [self.obs_name, self.act_name]
            self.value_out_keys = [f"{self.agent_spec.name}.q"]

            self.critic = Critic(
                self.cfg.critic, 
                self.agent_spec.n,
                self.agent_spec.observation_spec,
                self.agent_spec.action_spec
            ).to(self.device)
        
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic.lr)       
        self.critic_loss_fn = {"mse": F.mse_loss, "smooth_l1": F.smooth_l1_loss}[self.cfg.critic_loss]

    def __call__(self, tensordict: TensorDict, deterministic: bool=False) -> TensorDict:
        actor_output = self._call_actor(tensordict, self.actor_params)
        action_noise = (
            actor_output[self.act_name]
            .clone()
            .normal_(0, self.policy_noise)
            .clamp_(-self.noise_clip, self.noise_clip)
        )
        actor_output[self.act_name].add_(action_noise)
        actor_output["action"].batch_size = tensordict.batch_size
        tensordict.update(actor_output)
        return tensordict

    def _call_actor(self, tensordict: TensorDict, params: TensorDict):
        actor_input = tensordict.select(*self.policy_in_keys)
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]
        actor_output = vmap(self.actor, in_dims=(1, 0), out_dims=1)(actor_input, params)
        return actor_output

    def train_op(self, data: TensorDict):
        self.replay_buffer.extend(data.to("cpu").reshape(-1))

        if len(self.replay_buffer) < self.cfg.buffer_size:
            print(f"{len(self.replay_buffer)} < {self.cfg.buffer_size}")
            return {}
        
        infos_critic = []
        infos_actor = []

        with tqdm(range(1, self.gradient_steps+1)) as t:
            for gradient_step in t:

                transition = self.replay_buffer.sample(self.batch_size).to(self.device)

                state   = transition[self.state_name]
                actions_taken = transition[self.act_name]

                reward  = transition[("next", "reward", f"{self.agent_spec.name}.reward")]
                next_dones  = transition[("next", "done")].float().unsqueeze(-1)
                next_state  = transition[("next", self.state_name)]

                with torch.no_grad():
                    next_action: torch.Tensor = self._call_actor(
                        transition["next"], self.actor_target_params
                    )[self.act_name]

                    if self.target_noise > 0: # target smoothing
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

                qs = self.critic(state, actions_taken)
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

                        actor_output = self._call_actor(transition, self.actor_params)
                        actions_new = actor_output[self.act_name]

                        actor_losses = []
                        for a in range(self.agent_spec.n):
                            actions = actions_taken.clone()
                            actions[..., a, :] = actions_new[..., a, :]
                            qs = self.critic(state, actions)
                            q = torch.min(qs, dim=-1, keepdim=True).values
                            actor_losses.append(- q.mean())

                        actor_loss = torch.stack(actor_losses).sum()
                        self.actor_opt.zero_grad()
                        actor_loss.backward()
                        actor_grad_norm = nn.utils.clip_grad_norm_(
                            self.actor_opt.param_groups[0]["params"], self.cfg.max_grad_norm
                        )
                        self.actor_opt.step()

                        infos_actor.append(TensorDict({
                            "actor_loss": actor_loss,
                            "actor_grad_norm": actor_grad_norm,
                        }, []))
                    
                    with torch.no_grad():
                        soft_update_td(self.actor_target_params, self.actor_params, self.cfg.tau)
                        soft_update(self.critic_target, self.critic, self.cfg.tau)

                t.set_postfix({"critic_loss": critic_loss.item()})
        
        infos = {**torch.stack(infos_actor), **torch.stack(infos_critic)}
        infos = {k: torch.mean(v).item() for k, v in infos.items()}
        return infos

def soft_update_td(target_params: TensorDict, params: TensorDict, tau: float):
    for target_param, param in zip(target_params.values(True, True), params.values(True, True)):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

from .modules.networks import MLP, ENCODERS_MAP
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
        self.state_spec = state_spec
        self.num_critics = num_critics

        self.critics = nn.ModuleList([
            self._make_critic() for _ in range(self.num_critics)
        ])

    def _make_critic(self):
        if isinstance(self.state_spec, (BoundedTensorSpec, UnboundedTensorSpec)):
            action_dim = self.act_space.shape[-1]
            state_dim = self.state_spec.shape[-1]
            num_units = [
                action_dim * self.num_agents + state_dim, 
                *self.cfg["hidden_units"]
            ]
            base = MLP(num_units)
        elif isinstance(self.state_spec, CompositeSpec):
            encoder_cls = ENCODERS_MAP[self.cfg.attn_encoder]
            base = encoder_cls(CompositeSpec(self.state_spec))
        else:
            raise NotImplementedError
        
        v_out = nn.Linear(base.output_shape.numel(), self.num_agents)
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


def soft_update_params(target: TensorDict, source: TensorDict, tau: float):
    ...