import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from omni_drones.learning.common import MyBuffer, make_encoder
from omni_drones.learning.modules.distributions import MultiOneHotCategorical
from omni_drones.learning.modules.networks import MLP
from collections import defaultdict

from torchrl.envs.transforms import CatTensors

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

class TwoHot(D.distribution):
    def __init__(
        self, 
        logits: torch.Tensor, 
        low=-20.0, 
        high=20.0, 
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=logits.shape[-1]).to(logits.device)

    @property
    def mean(self):
        return torch.sum(self.probs * self.buckets, -1)

    @property
    def mode(self):
        return self.mean

    def log_prob(self, x):
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)


class SymlogTransform(D.Transform):
    def _call(self, x):
        return symlog(x)

    def _inverse(self, y):
        return symexp(y)


def rollout(module: TensorDictModule, tensordict: TensorDict):


    return 


def make_decoder(cfg, observation_spec):

    return


def lambda_return(
    
):
    ...


class SequenceModel(nn.Module):
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        action_dim: int,
        hidden_units: list,
        dynamics_pred_units: list
    ) -> None:
        super().__init__()
        self.inp_layers = MLP([stoch_dim + action_dim] + hidden_units, nn.LayerNorm)
        self.gru = nn.GRUCell(hidden_units[-1], deter_dim)
        self.dynamics_pred = MLP([deter_dim] + dynamics_pred_units + [stoch_dim], nn.LayerNorm)
    
    def forward(self, stoch, action, deter):
        x = torch.cat([stoch, action], dim=-1)
        x = self.inp_layers(x)
        x, deter = self.gru(x, deter)
        prior_logits = self.dynamics_pred(x)
        return prior_logits, deter

class ObsEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        deter_dim: int,
        stoch_dim: int,
        hidden_units: list
    ):
        self.encoder = encoder
        self.obs_out_layers = MLP(
            [encoder.output_shape.numel() + deter_dim] + hidden_units + [stoch_dim], 
            nn.LayerNorm
        )
    
    def forward(self, obs, deter):
        embed = self.encoder(obs)
        x = torch.cat([embed, deter], dim=-1)
        post_logits = self.obs_out_layers(x)
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


class DreamerPolicy():
    def __init__(
        self,
        cfg,
        agent_spec,
    ) -> None:
        self.cfg = cfg
        self.buffer = MyBuffer()

        self.obs_pred_dist = defaultdict()
        TwoHotSymlog = lambda logits: D.TransformedDistribution(TwoHot(logits), SymlogTransform())
        self.value_pred_dist = TwoHotSymlog
        self.reward_pred_dist = TwoHotSymlog
        
        stoch_dim = cfg.world_model.stoch_dim
        deter_dim = cfg.world_model.deter_dim
        discrete_dim = cfg.world_model.discrete_dim
        latent_dim = stoch_dim * discrete_dim + deter_dim
        action_dim = agent_spec.action_spec.shape[0]

        encoder = make_encoder(cfg.encoder, agent_spec.observation_spec)
        self.wm = nn.ModuleDict({
            "encoder": ObsEncoder(encoder, deter_dim, stoch_dim, cfg.world_model.post_hidden_units),
            "sequence": SequenceModel(
                deter_dim, 
                stoch_dim,
                action_dim,
                cfg.world_model.hidden_units, # cat([stoch, action]) -> gru_input
                cfg.world_model.prior_hidden_units
            ),
            "reward_predictor": MLP([latent_dim] + cfg.reward_predictor.hidden_units + [255], nn.LayerNorm),
            "discount_predictor": MLP([latent_dim] + cfg.discount_predictor.hidden_units + [1], nn.LayerNorm),
            "decoder": make_decoder(cfg.decoder, agent_spec.observation_spec),
        })
        self.task_actor = ...
        self.critic = MLP([latent_dim] + cfg.critic.hidden_units + [1], nn.LayerNorm)

        self.model_opt = torch.optim.Adam(self.wm.parameters(), self.world_model.lr)
        self.actor_opt = torch.optim.Adam(self.task_actor.parameters(), self.actor.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), self.critic.lr)

        self.obs_name = f"{self.agent_spec.name}.obs"
        self.policy = TensorDictSequential(
            TensorDictModule(self.wm["encoder"], [self.obs_name, "deter"], ["post_logits"]),
            TensorDictModule(Sample(MultiOneHotCategorical), ["post_logits"], ["stoch"]),
            CatTensors(["deter", "stoch"], ["latent"], del_keys=False),
            TensorDictModule(self.task_actor, ["latent"], ["action"]),
        )
        self.observe = TensorDictSequential(
            TensorDictModule(self.wm["encoder"], [self.obs_name, "deter"], ["post_logits"]),
            TensorDictModule(Sample(MultiOneHotCategorical), ["post_logits"], ["stoch"]),
            TensorDictModule(
                self.wm["sequence"], 
                ["stoch", "action", "deter"], 
                [("next", "prior_logits"), ("next", "deter")]
            ),
        )
        self.imagine = TensorDictSequential(
            CatTensors(["deter", "stoch"], ["latent"], del_keys=False),
            TensorDictModule(self.task_actor, ["latent"], ["action"]),
            TensorDictModule(
                self.wm["sequence"], 
                ["stoch", "action", "deter"], 
                [("next", "prior_logits"), ("next", "deter")]
            ),
            TensorDictModule(Sample(MultiOneHotCategorical), [("next","prior_logits")], [("next", "stoch")]),
        )

    def __call__(self, tensordict: TensorDict):
        tensordict = self.policy(tensordict)
        return tensordict
    
    def train_op(self, tensordict: TensorDict):
        self.buffer.extend(tensordict)

        metrics = defaultdict(list)

        for _ in range(self.cfg.gradient_steps):
            batch = self.buffer.sample()

            get_dist = lambda logits: MultiOneHotCategorical(logits=logits)
            batch = rollout(self.observe, batch)
            prior_logits = batch["prior_logits"]
            post_logits = batch["post_logits"]
            reward = batch["reward"]
            reward_pred = batch["reward_pred"]
            discount = 1. - batch[("next", "done")].float()
            discount_pred = batch["discount_pred"]

            model_losses = {}
            model_losses["dyn"] = D.kl_divergence(get_dist(post_logits.detach()), get_dist(prior_logits))
            model_losses["rep"] = D.kl_divergence(get_dist(post_logits), get_dist(prior_logits.detach()))
            model_losses["reward"] = self.reward_pred_dist(reward_pred).log_prob(reward)
            model_losses["discount"] = F.binary_cross_entropy_with_logits(discount_pred, discount)

            model_loss = sum(model_losses.values())
            self.model_opt.zero_grad()
            model_loss.backward()
            wm_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.wm.parameters(), self.cfg.max_grad_norm)
            self.model_opt.step()

            metrics["model_loss"].append(model_loss)
            metrics["wm_grad_norm"].append(wm_grad_norm)

            imag_traj = rollout(self.imagine, batch.reshape(-1))
            
            value_pred = self.value_pred_dist(imag_traj[""]).mean
            reward_pred = self.reward_pred_dist((imag_traj[""])).mean
            discount_pred = (imag_traj[""]).mean

            ret, weight = lambda_return(
                reward_pred,

            )
            offset, scale = self.retnorm(ret)
            normed_ret = (ret - offset) / scale
            normed_base = (value_pred - offset) / scale
            adv = normed_ret - normed_base

            policy = self.task_actor()
            log_prob = policy["action_log_prob"]
            entropy = policy["action_entropy"]
            if self.actor_grad == "reinforce":
                actor_loss = - log_prob * adv
            elif self.actor_grad == "dynamics":
                actor_loss = - adv
            else:
                raise NotImplementedError
            actor_loss -= self.cfg.actor_entropy * entropy
            actor_loss = torch.mean(weight * actor_loss)
            self.actor_opt.zero_grad()
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.task_actor.prameters())
            self.actor_opt.step()
            metrics["actor_loss"].append(actor_loss)
            metrics["actor_entropy"].append(entropy.mean())
            metrics["actor_grad_norm"].append(actor_grad_norm)


        metrics = TensorDict({
            k: torch.stack(v) for k, v in metrics.items()
        }, [])

        return metrics



