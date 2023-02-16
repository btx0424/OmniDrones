import torch
import torch.nn as nn
import torch.nn.functional as F
import functorch
import torchopt
import time
from torch.optim import lr_scheduler
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from typing import Any, Dict, List, Tuple, Union, Optional

from omni_drones.envs.isaac_env import AgentSpec

from .utils import valuenorm
from .utils.clip_grad import clip_grad_norm_
from .utils.gae import compute_gae, compute_gae_
from .common import ActorCriticPolicy

LR_SCHEDULER = lr_scheduler._LRScheduler

def linear_schedule(start_factor, end_factor, total_iters):
    start_factor = torch.as_tensor(start_factor)
    end_factor = torch.as_tensor(end_factor)
    total_iters = torch.as_tensor(total_iters)
    assert len(start_factor) == len(end_factor)
    assert (total_iters > 0).all()
    def get_weights(iter_):
        return torch.min(
            (end_factor-start_factor)/total_iters * iter_ + start_factor,
            end_factor
        )
    return get_weights

import functools
def with_extended_batch_size(func, batch_size):
    if isinstance(batch_size, int):
        batch_size = [batch_size]
    @functools.wraps(func)
    def func_with_extended_batch_size(tensordict: TensorDict, **kwargs):
        tensordict.batch_size = [*tensordict.batch_size, *batch_size]
        return func(tensordict, **kwargs)
    return func_with_extended_batch_size

class MAPPOPolicy(ActorCriticPolicy):
    """
    TODO: document the expected behaviors
    
    Args:
    
        critic_input: str, `'obs'` or `'state'`
            `'obs'`: critic output (R,)
            `'state'`: critic output (num_agents, R)
            where R is decided by `value_learning`

        value_learning: str, `'combined'`| `'separate'`
            `'combined'`: fit values.sum(-1) to returns.sum(-1)
            `'separate'`: fit values to returns
    """
    def __init__(self, 
        cfg,
        agent_type: AgentSpec,
        device="cuda"
    ) -> None:
        super().__init__(cfg, agent_type, device)

        self.clip_param = cfg.clip_param
        self.ppo_epoch = int(cfg.ppo_epochs)
        self.num_minibatches = int(cfg.num_minibatches)
        self.normalize_advantages = cfg.normalize_advantages

        self.entropy_coef = cfg.entropy_coef
        self.gae_gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda

        self.act_dim = agent_type.action_spec.shape.numel()

        if cfg.reward_weights is not None:
            from omegaconf.listconfig import ListConfig
            if isinstance(cfg.reward_weights, ListConfig):
                if not len(cfg.reward_weights) == self.agent_type.num_rewards:
                    raise RuntimeError(
                        f'AgentType(name="{agent_type.name}") has {agent_type.num_rewards} rewards'
                        f'but the given reward weights are {reward_weights}.'
                    )
                reward_weights = lambda step: torch.tensor(cfg.reward_weights, device=self.device).float() 
            elif isinstance(cfg.reward_weights, str):
                _reward_weights = eval(cfg.reward_weights)
                assert callable(_reward_weights)
                reward_weights = lambda step: torch.as_tensor(_reward_weights(step), device=self.device)
            else:
                raise ValueError(
                    f"{cfg.reward_weights} is of type {type(cfg.reward_weights)}"
                )
        else:
            reward_weights = lambda step: torch.ones(self.agent_type.reward_spec.shape.numel(), device=self.device)
        self.reward_weights = reward_weights
        
        self.train_in_keys = [
            self.obs_name,
            ("next", self.obs_name),
            self.act_name,
            self.act_logps_name,
            ("reward", f"{self.agent_name}.reward"),
            "done",
            f"{self.agent_name}.done",
            "state_value",
        ]

        self.in_keys = [self.obs_name]
        self.n_updates = 0

    @property
    def act_logps_name(self):
        return f"{self.agent_name}.action_logp"

    def make_model(self):
        self.make_actor()
        self.make_critic()

    def make_actor(self):
        cfg = self.cfg.actor

        self.actor_in_keys = [self.obs_name, self.act_name]
        self.actor_out_keys = [self.act_name, self.act_logps_name, f"{self.agent_name}.action_entropy"]
        self.actor_opt = torchopt.adam(lr=cfg.lr)

        def actor_loss(params, actor_input, advantages, log_probs_old):
            actor_output = fmodel(params, buffers, actor_input)
            log_probs = actor_output[self.act_logps_name]
            dist_entropy = actor_output[f"{self.agent_name}.action_entropy"]
            assert advantages.shape == log_probs.shape == dist_entropy.shape

            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages

            policy_loss = - torch.min(surr1, surr2) * self.act_dim
            policy_loss = torch.mean(policy_loss - self.entropy_coef * dist_entropy)

            log_ratio = (log_probs - log_probs_old)
            approx_kl = torch.mean(torch.exp(log_ratio) - 1 - log_ratio)
            clip_frac = torch.mean((torch.abs(ratio - 1) > self.clip_param).float())

            return policy_loss, (dist_entropy, approx_kl, clip_frac)
        
        self.share_actor = self.cfg.share_actor
        if self.share_actor:
            self.actor = TensorDictModule(
                make_ppo_actor(cfg, self.agent_type.observation_spec, self.agent_type.action_spec),
                in_keys=self.actor_in_keys, out_keys=self.actor_out_keys
            ).to(self.device)
            self.actor_func_module = (fmodel, params, buffers) = functorch.make_functional_with_buffers(self.actor)
            self.actor_func = functorch.vmap(fmodel, in_dims=(None, None, 1), out_dims=1, randomness="different")
            self.actor_opt_state = self.actor_opt.init(params)
            self.actor_loss = functorch.vmap(actor_loss, in_dims=(None, 1, 1, 1)) # for torch.autograd.grad
            self.actor_grad = functorch.vmap(
                functorch.grad_and_value(actor_loss, has_aux=True), in_dims=(None, 1, 1, 1))
        else:
            self.actor = nn.ModuleList([
                TensorDictModule(
                    make_ppo_actor(cfg, self.agent_type.observation_spec, self.agent_type.action_spec),
                    in_keys=self.actor_in_keys, out_keys=self.actor_out_keys
                ) for _ in range(self.num_agents)
            ]).to(self.device)
            self.actor_func_module = (fmodel, params, buffers) = functorch.combine_state_for_ensemble(self.actor)
            self.actor_func = functorch.vmap(fmodel, in_dims=(0, 0, 1), out_dims=1, randomness="different")
            self.actor_opt_state = functorch.vmap(self.actor_opt.init)(params)
            for param in params:
                param.requires_grad_(True)
            self.actor_loss = functorch.vmap(actor_loss, in_dims=(0, 1, 1, 1)) # for torch.autograd.grad
            self.actor_grad = functorch.vmap(
                functorch.grad_and_value(actor_loss, has_aux=True), in_dims=(0, 1, 1, 1))

    def make_critic(self):
        cfg = self.cfg.critic

        self.value_learning = self.cfg.value_learning
        assert self.value_learning in ["combined", "separate"]
        
        if cfg.use_huber_loss:
            self.critic_loss_fn = nn.HuberLoss(reduction="none", delta=cfg.huber_delta)
        else:
            self.critic_loss_fn = nn.MSELoss(reduction="none")
        
        self.critic_output_shape = (
            self.agent_type.n if self.cfg.critic_input == "state" else 1,
            self.agent_type.num_rewards if self.value_learning == "separate" else 1,
        )
        
        if self.cfg.critic_input == "state":
            if self.agent_type.state_spec is None:
                raise ValueError
            self.critic_in_keys = [f"{self.agent_name}.state"]
            self.critic_out_keys = ["state_value"]

            self.critic = TensorDictModule(
                CentralizedCritic(
                    cfg,
                    entity_ids=torch.arange(self.agent_type.n, device=self.device),
                    state_space=self.agent_type.state_space,
                    num_rewards=self.agent_type.num_rewards if self.value_learning == "separate" else 1,
                ),
                in_keys=self.critic_in_keys,
                out_keys=self.critic_out_keys
            ).to(self.device)
            self.value_func = self.critic
        elif self.cfg.critic_input == "obs":
            self.critic_in_keys = [f"{self.agent_name}.obs"]
            self.critic_out_keys = ["state_value"]

            self.critic = TensorDictModule(
                SharedCritic(
                    cfg,
                    self.agent_type.observation_spec,
                    self.agent_type.num_rewards if self.value_learning == "separate" else 1,
                ),
                in_keys=self.critic_in_keys, 
                out_keys=self.critic_out_keys
            ).to(self.device)
            self.value_func = with_extended_batch_size(
                functorch.vmap(self.critic, in_dims=1, out_dims=1),
                batch_size=self.num_agents
            )
        else:
            raise ValueError(self.cfg.critic_input)

        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = cfg.lr_scheduler
        if scheduler is not None:
            scheduler = eval(scheduler)
            self.critic_opt_scheduler: LR_SCHEDULER = scheduler(self.critic_opt, **cfg.lr_scheduler_kwargs)
        
        if hasattr(cfg, "value_norm") and cfg.value_norm is not None:
            # The original MAPPO implementation uses ValueNorm1 with a very large beta,
            # and normalizes advantages at batch level.
            # Tianshou (https://github.com/thu-ml/tianshou) uses ValueNorm2 with subtract_mean=False, 
            # and normalizes advantages at mini-batch level.
            # Empirically the performance is similar on most of the tasks.
            cls = getattr(valuenorm, cfg.value_norm["class"])
            self.value_normalizer: valuenorm.Normalizer = cls(
                input_shape=(
                    self.num_agents, 
                    self.agent_type.num_rewards if self.value_learning == "separate" else 1,
                ),
                **cfg.value_norm["kwargs"]
            ).to(self.device)
        
    def policy_op(self, tensordict: TensorDict, training: bool):
        fmodel, params, buffers = self.actor_func_module
        # TODO@btx0424 improve this behavior
        # set the right batch size at a proper place
        actor_input = tensordict.select(*self.actor_in_keys, strict=False)
        actor_input.batch_size = [*actor_input.batch_size, self.num_agents]
        tensordict = self.actor_func(params, buffers, actor_input, deterministic=(not training))
        return tensordict

    def value_op(self, tensordict: TensorDict, training: bool) -> TensorDict:
        critic_input = tensordict.select(*self.critic_in_keys)
        tensordict = self.value_func(critic_input)
        return tensordict

    def __call__(self, tensordict: TensorDict):
        
        actor_input = tensordict.select(*self.in_keys)
        tensordict.update(self.policy_op(actor_input, True))

        critic_input = tensordict.select(*self.in_keys)
        tensordict.update(self.value_func(critic_input))

        return tensordict

    def update_actor(self, batch: TensorDict) -> Dict[str, Any]:
        fmodel, params, buffers = self.actor_func_module
        advantages = batch["advantages"]
        actor_input = batch.select(*self.actor_in_keys)
        actor_input.batch_size = [*actor_input.batch_size, self.num_agents]
        policy_loss, (dist_entropy, approx_kl, clip_frac) = self.actor_loss(
            params, actor_input, advantages, batch[self.act_logps_name])
        if self.share_actor:
            grads = torch.autograd.grad(policy_loss.mean(), params)
            grad_norm = clip_grad_norm_(grads, max_norm=self.cfg.max_grad_norm)
            updates, self.actor_opt_state = self.actor_opt.update(grads, self.actor_opt_state)
        else:
            grads = torch.autograd.grad(policy_loss.sum(), params)
            grad_norm = functorch.vmap(clip_grad_norm_)(grads, max_norm=self.cfg.max_grad_norm)
            updates, self.actor_opt_state = functorch.vmap(self.actor_opt.update)(grads, self.actor_opt_state)
        torchopt.apply_updates(params, updates, inplace=True)

        return {
            "policy_loss": policy_loss.mean(), 
            "actor_grad_norm": grad_norm.mean(), 
            "dist_entropy": dist_entropy.mean(),
            "clip_fraction": clip_frac.mean(),
            "approx_kl": approx_kl.mean(),
        }

    def update_critic(self, batch: TensorDict) -> Dict[str, Any]:
        critic_input = batch.select(*self.critic_in_keys)
        values = self.value_func(critic_input)["state_value"]
        b_values = batch["state_value"]
        b_returns = batch["returns"]
        assert values.shape == b_values.shape == b_returns.shape
        value_pred_clipped = b_values + (values - b_values).clamp(-self.clip_param, self.clip_param)
                
        value_loss_clipped = self.critic_loss_fn(b_returns, value_pred_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        
        value_loss = torch.max(value_loss_original, value_loss_clipped)
        
        value_loss.sum(-1).mean().backward() # do not multiply weights here
        grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_opt.step()
        self.critic_opt.zero_grad(set_to_none=True)
        return {
            "value_loss": value_loss.mean(),
            "critic_grad_norm": grad_norm,
        }

    def _get_dones(self, tensordict: TensorDict):
        env_done = tensordict["done"].unsqueeze(-1)
        agent_done = tensordict.get(
            f"{self.agent_name}.done", 
            env_done.expand(*env_done.shape[:-2], self.num_agents, 1)
        )
        done = (agent_done | env_done)
        return done
    
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.select(*self.train_in_keys, strict=False)
        next_tensordict = tensordict["next"][:, -1]
        with torch.no_grad():
            value_output = self.value_func(next_tensordict)
        
        rewards = tensordict[("reward", f"{self.agent_name}.reward")]
        if self.value_learning == "combined":
            rewards = (rewards * self.reward_weights(self.n_updates)).sum(-1, keepdim=True)
        values = tensordict["state_value"]
        next_value = value_output["state_value"].squeeze(0)

        if hasattr(self, "value_normalizer"):
            values = self.value_normalizer.denormalize(values)
            next_value = self.value_normalizer.denormalize(next_value)
        
        dones = self._get_dones(tensordict)

        tensordict["advantages"], tensordict["returns"] = compute_gae(
            rewards, dones, values, next_value, 
            gamma=self.gae_gamma, lmbda=self.gae_lambda,
        )

        if self.value_learning == "separate":
            tensordict["advantages"] = (tensordict["advantages"] * self.reward_weights(self.n_updates)).sum(-1, keepdim=True)

        advantages_mean = tensordict["advantages"].mean()
        advantages_std = tensordict["advantages"].std()
        if self.normalize_advantages:
            tensordict["advantages"] = (tensordict["advantages"] - advantages_mean) / (advantages_std + 1e-8)
        
        if hasattr(self, "value_normalizer"):
            self.value_normalizer.update(tensordict["returns"])
            tensordict["returns"] = self.value_normalizer.normalize(tensordict["returns"])
        
        train_info = []
        for ppo_epoch in range(self.ppo_epoch):
            for minibatch in self.make_dataset(tensordict):
                train_info.append(TensorDict({
                    **self.update_actor(minibatch), 
                    **self.update_critic(minibatch)
                }, batch_size=[]))

        train_info: TensorDict = torch.stack(train_info)
        train_info = train_info.apply(lambda x: x.mean(0), batch_size=[])
        train_info["advantages_mean"] = advantages_mean
        train_info["advantages_std"] = advantages_std
        train_info["action_norm"] = tensordict[self.act_name].float().norm(dim=-1).mean()
        if hasattr(self, "value_normalizer"):
            train_info["value_running_mean"] = self.value_normalizer.running_mean.mean()
        self.n_updates += 1
        return {f"{self.agent_name}/{k}": v for k, v in train_info.items()}

    def train_on_batch(self, batch: TensorDict, next_tensordict: TensorDict):
        start = time.perf_counter()
        with torch.no_grad():
            value_output = self.value_op(next_tensordict, training=False)

        rewards = batch[("reward", f"{self.agent_name}.reward")]
        if self.value_learning == "combined":
            rewards = (rewards * self.reward_weights(self.n_updates)).sum(-1, keepdim=True)
        values = batch["state_value"]
        next_value = value_output["state_value"].squeeze(0)

        if hasattr(self, "value_normalizer"):
            values = self.value_normalizer.denormalize(values)
            next_value = self.value_normalizer.denormalize(next_value)
        
        dones = self._get_dones(batch)

        batch["advantages"], batch["returns"] = compute_gae_(
            reward=rewards,
            done=dones,
            value=values,
            next_value=next_value,
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
        )
        if self.value_learning == "separate":
            batch["advantages"] = (batch["advantages"] * self.reward_weights(self.n_updates)).sum(-1, keepdim=True)

        advantages_mean = batch["advantages"].mean()
        advantages_std = batch["advantages"].std()
        if self.normalize_advantages:
            batch["advantages"] = (batch["advantages"] - advantages_mean) / (advantages_std + 1e-8)
        
        if hasattr(self, "value_normalizer"):
            self.value_normalizer.update(batch["returns"])
            batch["returns"] = self.value_normalizer.normalize(batch["returns"])
        
        train_info = []
        for ppo_epoch in range(self.ppo_epoch):
            for minibatch in self.make_dataset(batch):
                train_info.append(TensorDict({
                    **self.update_actor(minibatch), 
                    **self.update_critic(minibatch)
                }, batch_size=[]))

        train_info: TensorDict = torch.stack(train_info)
        train_info = train_info.apply(lambda x: x.mean(0), batch_size=[])
        train_info["advantages_mean"] = advantages_mean
        train_info["advantages_std"] = advantages_std
        train_info["action_norm"] = batch[self.act_name].float().norm(dim=-1).mean()
        if hasattr(self, "value_normalizer"):
            train_info["value_running_mean"] = self.value_normalizer.running_mean.mean()
        self.n_updates += 1
        # print(f"training takes {time.perf_counter() - start} s")
        return {f"{self.agent_name}/{k}": v for k, v in train_info.items()}


    def make_dataset(self, tensordict: TensorDict):
        """
            tensordict: [T, E, A]
        """
        if False:
            assert tensordict["dones"].shape[0] % self.data_chunk_length == 0
            tensordict = tensordict.flatten(1, 2)
            chunk_len = self.data_chunk_length if self.data_chunk_length > 0 else tensordict["dones"].shape[0]
            batch_size = tensordict["dones"].shape[0] // chunk_len * tensordict["dones"].shape[1]
            tensordict = tensordict.unflatten(0, [-1, chunk_len]).transpose(0, 1).flatten(1, 2)
            perm = torch.randperm(batch_size, device=self.device).reshape(self.num_minibatches, -1)
            for indices in perm:
                yield tensordict[:, indices]
        else:
            tensordict = tensordict.reshape(-1)
            batch_size = (tensordict.shape.numel() // self.num_minibatches) * self.num_minibatches
            perm = torch.randperm(batch_size, device=self.device).reshape(self.num_minibatches, -1)
            for indices in perm:
                yield tensordict[indices]
        
    def state_dict(self):
        state_dict =  {
            "actor": self.actor.state_dict(),
            "actor_opt_state": self.actor_opt_state,
            "critic": self.critic.state_dict(),
            "critic_opt": self.critic_opt.state_dict()
        }
        if hasattr(self, "critic_opt_scheduler"):
            state_dict["critic_opt_scheduler"] = self.critic_opt_scheduler.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_opt_state = state_dict["actor_opt_state"]
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_opt.load_state_dict(state_dict["critic_opt"])


from .utils.network import ENCODERS_MAP, SplitEmbedding, MLP
from .utils.distributions import (
    MultiCategoricalModule,
    DiagGaussian, 
    IndependentBetaModule,
    IndependentNormalModule,
)

from torchrl.data import (
    TensorSpec, 
    BoundedTensorSpec, 
    UnboundedContinuousTensorSpec as UnboundedTensorSpec,
    CompositeSpec,
    MultiDiscreteTensorSpec
)

def make_ppo_actor(cfg, observation_spec: TensorSpec, action_spec: TensorSpec):

    if isinstance(observation_spec, (BoundedTensorSpec, UnboundedTensorSpec)):
        if not len(observation_spec.shape) == 1:
            raise ValueError
        input_dim = observation_spec.shape[0]
        base = nn.Sequential(
            nn.LayerNorm(input_dim),
            MLP([input_dim] + cfg.hidden_units),
        )
        base.output_shape = torch.Size((cfg.hidden_units[-1],))
    elif isinstance(observation_spec, CompositeSpec):
        encoder_cls = ENCODERS_MAP[cfg.attn_encoder]
        base = encoder_cls(observation_spec)
    else:
        raise NotImplementedError(observation_spec)
    
    if isinstance(action_spec, MultiDiscreteTensorSpec):
        act_dist = MultiCategoricalModule(base.output_shape.numel(), action_spec.nvec)
    elif isinstance(action_spec, (UnboundedTensorSpec, BoundedTensorSpec)):
        action_dim = action_spec.shape.numel()
        act_dist = DiagGaussian(base.output_shape.numel(), action_dim, True, 0.01)
        # act_dist = IndependentNormalModule(inputs_dim, action_dim, True)
        # act_dist = IndependentBetaModule(inputs_dim, action_dim)
    else:
        raise NotImplementedError(action_spec)
    
    return StochasticActor(base, act_dist)


class StochasticActor(nn.Module):
    def __init__(self, 
        base: nn.Module,
        act_dist: nn.Module,
        rnn: Optional[nn.Module]=None,
    ) -> None:
        super().__init__()
        self.base = base
        self.act_dist = act_dist
        if rnn is not None:
            self.rnn = rnn
    
    def forward(self, 
        obs: Union[torch.Tensor, TensorDict], 
        action: torch.Tensor=None,
        done=None, 
        rnn_state=None, 
        deterministic=False
    ):
        actor_features = self.base(obs)
        if hasattr(self, "rnn"):
            pass
        action_dist = self.act_dist(actor_features)
        if action is None:
            action = action_dist.mode if deterministic else action_dist.sample()
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)
            return action, action_log_probs
        else:
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)
            dist_entropy = action_dist.entropy().unsqueeze(-1)
            return action, action_log_probs, dist_entropy


import math
class SharedCritic(nn.Module):
    def __init__(self, 
        args, 
        observation_spec: TensorSpec,
        output_shape: Tuple[int, ...],
    ):
        super().__init__()
        if type(output_shape) == int: 
            output_shape = (output_shape,)
        self.output_shape = torch.Size(output_shape)

        if isinstance(observation_spec, (BoundedTensorSpec, UnboundedTensorSpec)):
            input_dim = observation_spec.shape.numel()
            self.base = nn.Sequential(
                nn.LayerNorm(input_dim),
                MLP([input_dim] + args.hidden_units)
            )
            self.base.output_shape = torch.Size((args.hidden_units[-1],))
        elif isinstance(observation_spec, CompositeSpec):    
            encoder_cls = ENCODERS_MAP[args.attn_encoder]
            self.base = encoder_cls(observation_spec)
        else:
            raise TypeError(observation_spec)

        self.v_out = nn.Linear(self.base.output_shape.numel(), self.output_shape.numel())
        nn.init.orthogonal_(self.v_out.weight, args.gain)

    def forward(self, obs: torch.Tensor):
        critic_features = self.base(obs)
        values = self.v_out(critic_features)
        if len(self.output_shape) > 1:
            values = values.unflatten(-1, self.output_shape)
        return values

INDEX_TYPE = Union[int, slice, torch.LongTensor, List[int]]
class CentralizedCritic(nn.Module):
    """Critic for centralized training.
    
    Args:
        entity_ids: indices of the entities that are considered as agents.
        
    """
    def __init__(self,
        entity_ids: INDEX_TYPE,
        state_spec: CompositeSpec, 
        num_rewards: int,
        embed_dim = 128,
        nhead=1,
        num_layers=1,
    ):
        super().__init__()
        self.entity_ids = entity_ids
        self.embed = SplitEmbedding(state_spec, embed_dim=embed_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.v_out = nn.Linear(embed_dim, num_rewards)
    
    def forward(self, x: torch.Tensor):
        x = self.embed(x)
        x = self.encoder(x)
        values = self.v_out(x[..., self.entity_ids, :])
        return values

