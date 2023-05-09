import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from omni_drones.learning.common import MyBuffer, make_encoder
from omni_drones.learning.modules.distributions import (
    MultiOneHotCategorical, 
    IndependentNormal,
    TwoHot,
)
from omni_drones.learning.modules.networks import MLP
from collections import defaultdict

from torchrl.envs.utils import step_mdp
from torchrl.envs.transforms import CatTensors
from torchrl.data import UnboundedContinuousTensorSpec, BoundedTensorSpec
from torchrl.modules.distributions import NormalParamWrapper

from tqdm import tqdm

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RewardEMA(nn.Module):
    def __init__(self, alpha=1e-2) -> None:
        super().__init__()
        self.alpha = alpha
        self.register_buffer("values", torch.zeros(2))
        self.register_buffer("range", torch.tensor([0.05, 0.95]))

        self.values: torch.Tensor
        self.range: torch.Tensor

    def forward(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values.mul_(1 - self.alpha).add_(self.alpha * x_quantile)
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset, scale


class Bernouli(D.Bernoulli):
    def __init__(self, logits):
        super().__init__(logits=logits)

    @property
    def mode(self):
        _mode = torch.round(self.mean)
        return (_mode.detach() + self.mean - self.mean.detach()).squeeze(-1)

    def log_prob(self, x):
        log_probs0 = -F.softplus(self.logits)
        log_probs1 = -F.softplus(-self.logits)

        return log_probs0 * (1 - x) + log_probs1 * x


def rollout(module: TensorDictModule, tensordict: TensorDict):
    tensordict = tensordict.select(*module.in_keys, "is_init", "prior_logits")
    tensordict_out = []
    *batch, time_steps = tensordict.shape
    _tensordict = tensordict[..., 0]

    update_values = tensordict.exclude(*module.out_keys)
    for t in range(time_steps):
        if "is_init" in _tensordict.keys():
            mask = 1. - _tensordict.get("is_init").float()
            _tensordict["deter"] = _tensordict["deter"] *  mask[..., None]
            _tensordict["prior_logits"] = _tensordict["prior_logits"] * mask[..., None, None]
        module(_tensordict)
        tensordict_out.append(_tensordict)
        if t < time_steps - 1:
            _tensordict = step_mdp(
                _tensordict.select(*module.out_keys, strict=False), keep_other=False
            )
            _tensordict = update_values[..., t + 1].update(_tensordict)

    return torch.stack(tensordict_out, tensordict.ndimension() - 1).contiguous()


def make_decoder(cfg, observation_spec, latent_dim):
    if isinstance(observation_spec, (UnboundedContinuousTensorSpec, BoundedTensorSpec)):
        decoder = nn.Sequential(
            MLP([latent_dim] + cfg.hidden_units, nn.LayerNorm),
            nn.Linear(cfg.hidden_units[-1], observation_spec.shape[-1])
        )
    else:
        pass
    return decoder


def lambda_return(
    reward,
    next_value,
    discount,
    lmbda: float
):
    adv = torch.zeros_like(reward)
    interm = reward + discount * next_value * (1 - lmbda)
    val = next_value[-1]
    for t in reversed(range(reward.shape[0])):
        adv[t] = val = interm[t] + discount[t] * lmbda * val
    weight = (torch.cumprod(discount, 0) / discount[0]).detach()
    return adv, weight


class SequenceModel(nn.Module):
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        discrete_dim: int,
        action_dim: int,
        hidden_units: list,
        dynamics_pred_units: list
    ) -> None:
        super().__init__()
        self.inp_layers = MLP([stoch_dim * discrete_dim + action_dim] + hidden_units, nn.LayerNorm)
        self.gru = nn.GRUCell(hidden_units[-1], deter_dim)
        self.dynamics_pred = MLP([deter_dim] + dynamics_pred_units + [stoch_dim * discrete_dim], nn.LayerNorm)
        self.stat_shape = (stoch_dim, discrete_dim)

    def forward(self, stoch, action, deter):
        x = torch.cat([stoch, action], dim=-1)
        x = self.inp_layers(x)
        x_bs = x.shape[:-1]
        deter = self.gru(x.flatten(0, -2), deter.flatten(0, -2)).unflatten(0, x_bs)
        prior_logits = self.dynamics_pred(deter).unflatten(-1, self.stat_shape)
        return prior_logits, deter


class ObsEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        deter_dim: int,
        output_shape: torch.Size,
        hidden_units: list
    ):
        super().__init__()
        self.encoder = encoder
        self.output_shape = torch.Size(output_shape)
        input_dim = encoder.output_shape.numel() + deter_dim
        self.obs_stat_layers = MLP(
            [input_dim] + hidden_units + [self.output_shape.numel()], 
            nn.LayerNorm
        )
    
    def forward(self, obs, deter):
        embed = self.encoder(obs)
        x = torch.cat([embed, deter], dim=-1)
        post_logits = self.obs_stat_layers(x).unflatten(-1, self.output_shape)
        return post_logits


class Sample(nn.Module):
    def __init__(self, dist_cls) -> None:
        super().__init__()
        self.dist_cls = dist_cls
    
    def forward(self, *args):
        dist = self.dist_cls(*args)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        entropy = dist.entropy()
        return sample, log_prob, entropy


class Actor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_units: list,
    ) -> None:
        super().__init__()
        self.layers = NormalParamWrapper(
            MLP([input_dim] + hidden_units + [action_dim * 2], nn.LayerNorm)
        )
    
    def forward(self, x):
        loc, scale = self.layers(x.detach())
        dist = IndependentNormal(loc, scale)
        samples = dist.rsample()
        log_prob = dist.log_prob(samples).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return samples, log_prob, entropy


class TwoHotSymlog(TwoHot):
    
    @property
    def mean(self):
        return symexp(super().mean)
    
    def log_prob(self, x):
        return super().log_prob(symlog(x))


class DreamerPolicy():
    def __init__(
        self,
        cfg,
        agent_spec,
        device
    ) -> None:
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        self.obs_pred_dist = lambda loc: D.Independent(D.Normal(loc, 1), 1)
        self.discount_pred_dist = lambda logits: D.Independent(Bernouli(logits), 1)
        
        stoch_dim = cfg.world_model.stoch_dim
        deter_dim = cfg.world_model.deter_dim
        discrete_dim = cfg.world_model.discrete_dim
        latent_dim = stoch_dim * discrete_dim + deter_dim
        action_dim = agent_spec.action_spec.shape[-1]

        encoder = make_encoder(cfg.encoder, agent_spec.observation_spec)
        self.wm = nn.ModuleDict({
            "encoder": ObsEncoder(
                encoder, deter_dim, (stoch_dim, discrete_dim), cfg.world_model.post_hidden_units
            ),
            "sequence": SequenceModel(
                deter_dim, 
                stoch_dim,
                discrete_dim,
                action_dim,
                cfg.world_model.hidden_units, # cat([stoch, action]) -> gru_input
                cfg.world_model.prior_hidden_units
            ),
            "discount_pred": MLP([latent_dim] + cfg.discount_pred.hidden_units + [1], nn.LayerNorm),
            "decoder": make_decoder(cfg.decoder, agent_spec.observation_spec, latent_dim),
        }).to(self.device)
        if self.cfg.reward_pred.dist == "TwoHotSymlog":
            self.value_pred_dist = lambda logits: TwoHotSymlog(logits=logits)
            self.wm["reward_pred"] = MLP([latent_dim] + cfg.reward_pred.hidden_units + [255], nn.LayerNorm).to(self.device)
        else:
            self.value_pred_dist = lambda loc: D.Independent(D.Normal(loc, 1), 1)
            self.wm["reward_pred"] = MLP([latent_dim] + cfg.reward_pred.hidden_units + [1], nn.LayerNorm).to(self.device)

        self.task_actor = Actor(latent_dim, action_dim, self.cfg.actor.hidden_units).to(self.device)
        if self.cfg.critic.dist == "TwoHotSymlog":
            self.reward_pred_dist = lambda logits: TwoHotSymlog(logits=logits)
            self.critic = MLP([latent_dim] + cfg.critic.hidden_units + [255], nn.LayerNorm).to(self.device)
        else:
            self.reward_pred_dist = lambda loc: D.Independent(D.Normal(loc, 1), 1)
            self.critic = MLP([latent_dim] + cfg.critic.hidden_units + [1], nn.LayerNorm).to(self.device)
        
        self.model_opt = torch.optim.Adam(self.wm.parameters(), self.cfg.world_model.lr)
        self.actor_opt = torch.optim.Adam(self.task_actor.parameters(), self.cfg.actor.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), self.cfg.critic.lr)

        self.retnorm = RewardEMA().to(self.device)

        self.obs_name = f"{self.agent_spec.name}.obs"
        self.action_name = ("action", f"{self.agent_spec.name}.action")
        sample_stoch = Sample(MultiOneHotCategorical)

        self.observe_step = TensorDictSequential(
            TensorDictModule(self.wm["encoder"], [self.obs_name, "deter"], ["post_logits"]),
            TensorDictModule(sample_stoch, ["post_logits"], ["stoch"]),
            TensorDictModule(nn.Flatten(-2), ["stoch"], ["stoch"]),
            CatTensors(["deter", "stoch"], "latent", del_keys=False),
        )
        self.imagine_step = TensorDictModule(
            self.wm["sequence"], 
            ["stoch", self.action_name, "deter"], 
            [("next", "prior_logits"), ("next", "deter")]
        )
        self.policy = TensorDictModule(self.task_actor, ["latent"], [self.action_name, "log_prob", "entropy"])

        self.observe = TensorDictSequential(
            self.observe_step,
            self.imagine_step
        )
        self.imagine = TensorDictSequential(
            CatTensors(["deter", "stoch"], "latent", del_keys=False),
            self.policy,
            self.imagine_step,
            TensorDictModule(sample_stoch, [("next","prior_logits")], [("next", "stoch")]),
            TensorDictModule(nn.Flatten(-2), [("next", "stoch")], [("next", "stoch")]),
            # CatTensors([("next", "deter"), ("next", "stoch")], ("next", "latent"), del_keys=False),
        )

        self.buffer = MyBuffer(cfg.buffer_size, device=self.device)


    def __call__(self, tensordict: TensorDict):
        if "deter" not in tensordict.keys():
            logits_shape = (self.cfg.world_model.stoch_dim, self.cfg.world_model.discrete_dim)
            tensordict.update({
                "deter": torch.zeros(*tensordict.batch_size, 1, self.cfg.world_model.deter_dim, device=tensordict.device),
                "prior_logits": torch.zeros(*tensordict.batch_size, 1, *logits_shape, device=tensordict.device)
            })
        mask = 1. - tensordict.get("is_init").float()
        tensordict["deter"] = tensordict["deter"] *  mask[..., None]
        tensordict["prior_logits"] = tensordict["prior_logits"] * mask[..., None, None]
        tensordict = self.observe_step(tensordict)
        tensordict = self.policy(tensordict)
        tensordict = self.imagine_step(tensordict)
        return tensordict
    
    def train_op(self, tensordict: TensorDict):
        self.buffer.extend(tensordict)
        if len(self.buffer) <= self.cfg.batch_length:
            return {}
        
        metrics = defaultdict(list)
        for _ in tqdm(range(self.cfg.gradient_steps)):
            batch = self.buffer.sample(self.cfg.batch_size, self.cfg.batch_length)
            batch = batch.exclude("collector", "progress")
            batch.batch_size = [*batch.batch_size, 1]
            batch = batch.squeeze(2)

            reward = batch[("next", "reward", f"{self.agent_spec.name}.reward")]
            discount = (1. - batch[("next", "done")].float()).unsqueeze(-1)

            get_dist = lambda logits: MultiOneHotCategorical(logits=logits)
            batch = rollout(self.observe, batch)
            prior_logits = batch["prior_logits"]
            post_logits = batch["post_logits"]
            latent = batch["latent"]
            
            reward_pred = self.reward_pred_dist(self.wm["reward_pred"](latent))
            discount_pred = self.discount_pred_dist(self.wm["discount_pred"](latent))
            obs_pred = self.obs_pred_dist(self.wm["decoder"](latent))

            model_losses = {}
            model_losses["dyn"] = (
                D.kl_divergence(get_dist(post_logits.detach()), get_dist(prior_logits))
                .clip(self.cfg.world_model.kl_free)
                .mean()
            )
            model_losses["rep"] = (
                D.kl_divergence(get_dist(post_logits), get_dist(prior_logits.detach()))
                .clip(self.cfg.world_model.kl_free)
                .mean()
            )
            # model_losses["reward"] = - torch.mean(reward_pred.log_prob(reward))
            model_losses["reward"] = F.mse_loss(reward_pred.base_dist.loc, reward)
            model_losses["discount"] = - torch.mean(discount_pred.log_prob(discount))
            model_losses["obs"] = - torch.mean(obs_pred.log_prob(batch[self.obs_name]))

            model_loss = torch.mean(sum(model_losses.values()))
            self.model_opt.zero_grad()
            model_loss.backward()
            wm_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.wm.parameters(), self.cfg.max_grad_norm)
            self.model_opt.step()

            metrics["model_loss"].append(model_loss)
            metrics["wm_grad_norm"].append(wm_grad_norm)
            metrics["model_loss/dyn"].append(model_losses["dyn"])
            metrics["model_loss/rep"].append(model_losses["rep"])
            metrics["model_loss/rew"].append(model_losses["reward"])
            metrics["model_loss/cont"].append(model_losses["discount"])
            metrics["model_loss/obs"].append(model_losses["obs"])

            imag_traj = []
            _tensordict = batch.reshape(-1).select(*self.imagine.in_keys).detach()
            for t in range(self.cfg.imag_horizon):
                _tensordict = self.imagine(_tensordict)
                imag_traj.append(_tensordict)
                _tensordict = step_mdp(_tensordict, keep_other=False)
            imag_traj = torch.stack(imag_traj).contiguous()

            latent = imag_traj["latent"]
            value_pred = self.value_pred_dist(self.critic(latent)).mean
            reward_pred = self.reward_pred_dist(self.wm["reward_pred"](latent)).mean
            discount_pred = self.discount_pred_dist(self.wm["discount_pred"](latent)).mean

            ret, weight = lambda_return(
                reward_pred[:-1],
                value_pred[1:],
                discount_pred[:-1] * self.cfg.discount,
                self.cfg.return_lambda
            )
            offset, scale = self.retnorm(ret)
            normed_ret = (ret - offset) / scale
            normed_base = (value_pred[:-1] - offset) / scale
            adv = normed_ret - normed_base

            _, log_prob, entropy = self.task_actor(latent[:-1].detach())
            if self.cfg.actor_grad == "reinforce":
                actor_loss = - log_prob * adv
            elif self.cfg.actor_grad == "dynamics":
                actor_loss = - adv
            else:
                raise NotImplementedError
            actor_loss -= self.cfg.actor_entropy_coeff * entropy
            actor_loss = torch.mean(weight * actor_loss)
            self.actor_opt.zero_grad()
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.task_actor.parameters(), self.cfg.max_grad_norm)
            self.actor_opt.step()
            metrics["actor_loss"].append(actor_loss)
            metrics["actor_entropy"].append(entropy.mean())
            metrics["actor_grad_norm"].append(actor_grad_norm)

            value_pred_dist = self.value_pred_dist(self.critic(latent[:-1].detach()))
            critic_loss = - value_pred_dist.log_prob(ret.detach()).unsqueeze(-1)
            critic_loss = torch.mean(weight * critic_loss)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
            self.critic_opt.step()
            metrics["critic_loss"].append(critic_loss)
            metrics["critic_grad_norm"].append(critic_grad_norm)


        metrics = {
            k: torch.stack(v).mean().item() 
            for k, v in metrics.items()
        }

        return metrics


