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

import copy
from tqdm import tqdm
from omni_drones.envs.isaac_env import AgentSpec
from tensordict import TensorDict
from .common import soft_update

class MASACPolicy(object):

    def __init__(self,
        cfg,
        agent_spec: AgentSpec,
        device: str="cuda",
    ) -> None:
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        if cfg.reward_weights is not None:
            reward_weights = torch.as_tensor(cfg.reward_weights, device=self.device).float() 
        else:
            reward_weights = torch.ones(self.agent_spec.reward_spec.shape, device=self.device)
        self.reward_weights = reward_weights

        self.gradient_steps = int(cfg.gradient_steps)
        self.batch_size = int(cfg.batch_size)

        self.obs_name = f"{self.agent_spec.name}.obs"
        self.act_name = f"{self.agent_spec.name}.action"
        if agent_spec.state_spec is not None:
            self.state_name = f"{self.agent_spec.name}.state"
        else:
            self.state_name = f"{self.agent_spec.name}.obs"
        self.reward_name = f"{self.agent_spec.name}.reward"

        self.make_actor()
        self.make_critic()
        
        self.action_dim = self.agent_spec.action_spec.shape.numel()
        self.target_entropy = - torch.tensor(self.action_dim, device=self.device)
        init_entropy = 1.0
        self.log_alpha = nn.Parameter(torch.tensor(init_entropy, device=self.device).log())
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)

        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=self.cfg.buffer_size),
            sampler=RandomSampler(),
        )
    
    def make_actor(self):

        self.policy_in_keys = [self.obs_name]
        self.policy_out_keys = [self.act_name, f"{self.agent_spec.name}.logp"]
        
        if self.cfg.share_actor:
            self.actor = TensorDictModule(
                Actor(
                    self.cfg, 
                    self.agent_spec.observation_spec, 
                    self.agent_spec.action_spec
                ),
                in_keys=self.policy_in_keys, out_keys=self.policy_out_keys
            ).to(self.device)
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        else:
            raise NotImplementedError


    def make_critic(self):
        if self.agent_spec.state_spec is not None:
            self.value_in_keys = [self.state_name, self.act_name]
            self.value_out_keys = [f"{self.agent_spec.name}.q"]

            self.critic = Critic(
                self.cfg, 
                self.agent_spec.state_spec,
                self.agent_spec.action_spec
            ).to(self.device)
        else:
            self.value_in_keys = [self.obs_name, self.act_name]
            self.value_out_keys = [f"{self.agent_spec.name}.q"]

            self.critic = Critic(
                self.cfg, 
                self.agent_spec.n,
                self.agent_spec.observation_spec,
                self.agent_spec.action_spec
            ).to(self.device)
        
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)       
        self.critic_loss_fn = F.mse_loss
        
        self.value_func = functorch.vmap(
            TensorDictModule(
                self.critic,
                in_keys=self.value_in_keys,
                out_keys=self.value_out_keys,
            ), 
            in_dims=1, out_dims=1
        )

    
    def __call__(self, tensordict: TensorDict, deterministic: bool=False) -> TensorDict:
        actor_input = tensordict.select(*self.policy_in_keys)
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]
        actor_output = self.actor(actor_input)
        tensordict.update(actor_output)
        return tensordict

    def train_op(self, data: TensorDict):
        self.replay_buffer.extend(data.reshape(-1))

        if len(self.replay_buffer) < self.batch_size:
            print(f"{len(self.replay_buffer)} < {self.batch_size}")
            return {}
        
        infos_critic = []
        infos_actor = []

        with tqdm(range(1, self.gradient_steps+1)) as t:
            for gradient_step in t:

                transition = self.replay_buffer.sample(self.batch_size)

                obs     = transition[self.obs_name]
                state   = transition[self.state_name]
                actions = transition[self.act_name]

                reward  = transition[("next", "reward", f"{self.agent_spec.name}.reward")]
                next_dones  = transition[("next", "done")].float()
                next_obs    = transition[("next", self.obs_name)]
                next_state  = transition[("next", self.state_name)]


                for i in range(self.agent_spec.n):
                    # TODO@btx0424 optimize this part
                    with torch.no_grad():
                        next_act, next_logp = self.actor(next_obs[:, i], deterministic=False)
                        next_qs = self.critic_target(next_state, next_act)
                        next_q = torch.min(next_qs, dim=-1)[0]
                        next_q = next_q - self.log_alpha.exp() * next_logp
                        target_q = (reward[:, i] + self.cfg.gamma * (1 - next_dones[:, i]) * next_q).detach()
                        assert not torch.isinf(target_q).any()
                        assert not torch.isnan(target_q).any()

                    qs = self.critic(state[:, i], actions[:, i])
                    critic_loss = sum(F.mse_loss(q, target_q) for q in qs.unbind(-1))
                    self.critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                    self.critic_opt.step()
                    infos_critic.append(TensorDict({
                        "critic_loss": critic_loss,
                        "critic_grad_norm": critic_grad_norm
                    }))

                    if (gradient_step + 1) % self.cfg.actor_delay == 0:

                        with freeze(self.critic):
                            act, logp = self.actor(obs[:, i], deterministic=False)
                            qs = self.critic(state[:, i], act)
                            q = torch.min(qs, dim=-1)[0]
                            actor_loss = (self.log_alpha.exp() * logp - q).mean()
                            self.actor_opt.zero_grad()
                            actor_loss.backward()
                            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                            self.actor_opt.step()

                            self.alpha_opt.zero_grad()
                            alpha_loss = (self.log_alpha * (-logp - self.target_entropy).detach()).mean()
                            alpha_loss.backward()
                            self.alpha_opt.step()

                            infos_actor.append(TensorDict({
                                "actor_loss": actor_loss,
                                "actor_grad_norm": actor_grad_norm,
                                "alpha": self.log_alpha.exp().detach(),
                                "alpha_loss": alpha_loss,
                            }))

                if (gradient_step + 1) % self.cfg.target_update_interval == 0:
                    soft_update(self.critic_target, self.critic, self.cfg.tau)

        return {**torch.stack(infos_actor).mean(), **torch.stack(infos_critic).mean()}

from .modules.networks import MLP
from .modules.distributions import TanhIndependentNormalModule

class Actor(nn.Module):
    def __init__(self, 
        cfg,
        observation_spec: TensorSpec, 
        action_spec: BoundedTensorSpec,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        if isinstance(observation_spec, (BoundedTensorSpec, UnboundedTensorSpec)):
            num_units = [observation_spec.shape.numel(), *cfg["hidden_units"]]
            self.base = MLP(num_units)
        elif isinstance(observation_spec, CompositeSpec):
            ...
        else:
            raise NotImplementedError(observation_spec)
        
        self.feature_norm = nn.LayerNorm(observation_spec.shape.numel())
        
        self.act = TanhIndependentNormalModule(
            cfg["hidden_units"][-1], 
            action_spec.shape.numel(), 
        )

    def forward(self, obs: torch.Tensor, deterministic: bool=False):
        if hasattr(self, "feature_norm"):
            obs = self.feature_norm(obs)
            
        x = self.base(obs)
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
            action_dim = self.act_space.shape.numel()
            state_dim = self.state_space.shape.numel()
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


import contextlib

@contextlib.contextmanager
def freeze(module: nn.Module):
    """Freeze the parameters of a module."""
    requires_grad = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, r in zip(module.parameters(), requires_grad):
            p.requires_grad_(r)

