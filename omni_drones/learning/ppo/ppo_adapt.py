import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential, TensorDictModuleBase

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Any, Mapping, Union, Tuple

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import GAE

@dataclass
class PPOConfig:
    name: str = "ppo_adaptive"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16

    checkpoint_path: Union[str, None] = None
    phase: str = "encoder"
    condition_mode: str = "cat"

    # what the adaptation module learns to predict
    adaptation_key: Any = "context"
    adaptation_loss: str = "mse"

    def __post_init__(self):
        assert self.condition_mode.lower() in ("cat", "film")
        assert self.adaptation_key in ("context", ("agents", "intrinsics"), "_feature")
        assert self.phase in ("encoder", "adaptation", "joint", "finetune")
        assert self.adaptation_loss.lower() in ("mse", "gan", "lsgan")

cs = ConfigStore.instance()
cs.store("ppo_adapt", node=PPOConfig, group="algo")
cs.store("ppo_adapt_latent_mse", node=PPOConfig(adaptation_loss="mse"), group="algo")
cs.store("ppo_adapt_latent_gan", node=PPOConfig(adaptation_loss="gan"), group="algo")
cs.store("ppo_adapt_latent_lsgan", node=PPOConfig(adaptation_loss="lsgan"), group="algo")
cs.store("ppo_adapt_raw", node=PPOConfig(adaptation_key=("agents", "intrinsics")), group="algo")


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


class TConv(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.tconv = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=1), nn.ELU(),
            nn.LazyConv1d(64, kernel_size=7, stride=2), nn.ELU(),
            nn.LazyConv1d(64, kernel_size=5, stride=2), nn.ELU(),
        )
        self.mlp = make_mlp([256, 256])
        self.out = nn.LazyLinear(out_dim)
    
    def forward(self, features: torch.Tensor):
        batch_shape = features.shape[:-2]
        features = features.flatten(0, -3) # [*, D, T]
        features_tconv = self.tconv(features).flatten(1)
        features = torch.cat([features_tconv, features[:, :, -1]], dim=1)
        features = self.mlp(features)
        features = self.out(features)
        return features.unflatten(0, batch_shape)

class TConvG(TConv):
    """
    A stochastic generator version for the adversarial adaptation modules.
    """
    def forward(self, features: torch.Tensor):
        batch_shape = features.shape[:-2]
        features = features.flatten(0, -3) # [*, D, T]
        features = torch.cat([
            self.tconv(features).flatten(1), 
            features[:, :, -1],
            torch.randn(features.shape[0], 32, device=features.device)
        ], dim=1)
        features = self.mlp(features)
        features = self.out(features)
        return features.unflatten(0, batch_shape)


class FiLM(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.f = nn.LazyLinear(feature_dim * 2)
        self.act = nn.ELU()
        self.ln = nn.LayerNorm(feature_dim)
    
    def forward(self, feature, context):
        w, b = self.f(context).chunk(2, dim=-1)
        feature = self.act(w * feature + b) + feature
        return feature


class PPOAdaptivePolicy(TensorDictModuleBase):
    
    def __init__(self,
        cfg: PPOConfig, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = 0.001
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.adaptation_key = self.cfg.adaptation_key
        self.phase = self.cfg.phase

        if not isinstance(self.adaptation_key, str):
            self.adaptation_key = tuple(self.adaptation_key)
        self.gae = GAE(0.99, 0.95)
        
        self.n_agents, self.action_dim = action_spec.shape[-2:]

        intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]

        fake_input = observation_spec.zero()

        self.encoder = TensorDictModule(
            nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])), 
            [("agents", "intrinsics")], ["context"]
        ).to(self.device)

        if self.cfg.condition_mode == "cat":
            condition = lambda: CatTensors(["_feature", "context"], "_feature", del_keys=False)
        elif self.cfg.condition_mode == "film":
            condition = lambda: TensorDictModule(FiLM(128), ["_feature", "context"], ["_feature"])

        actor_module = TensorDictSequential(
            TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["_feature"]),
            condition(),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 256]), Actor(self.action_dim)), 
                ["_feature"], ["loc", "scale"]
            )
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic = TensorDictSequential(
            TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["_feature"]),
            condition(),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)), 
                ["_feature"], ["state_value"]
            )
        ).to(self.device)
        
        self.value_norm = ValueNorm1(reward_spec.shape[-2:]).to(self.device)
        
        self.encoder(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
            
            self.actor.apply(init_)
            self.critic.apply(init_)
            self.encoder.apply(init_)

        if self.phase in ("adpatation", "finetune"):
            if self.cfg.adaptation_loss == "mse":
                self.adaptation_module = TensorDictModule(
                    TConv(fake_input[self.adaptation_key].shape[-1]), 
                    [("agents", "observation_h")], [self.adaptation_key]
                ).to(self.device)
                self.adaptation_module(fake_input)
                self.adaptation_loss = MSE(
                    self.adaptation_module, 
                    self.adaptation_key, 
                ).to(self.device)
            elif self.cfg.adaptation_loss == "gan":
                self.adaptation_module = TensorDictModule(
                    TConvG(fake_input[self.adaptation_key].shape[-1]), 
                    [("agents", "observation_h")], [self.adaptation_key]
                ).to(self.device)
                self.adaptation_module(fake_input)
                discriminator = TensorDictSequential(
                    TensorDictModule(TConv(128), [("agents", "observation_h")], ["condition"]),
                    CatTensors([self.adaptation_key, "condition"], "condition", del_keys=False),
                    TensorDictModule(
                        nn.Sequential(make_mlp([256]), nn.LazyLinear(1)),
                        ["condition"], ["label"]
                    )
                )
                self.adaptation_loss = GAN(
                    self.adaptation_module, 
                    discriminator,
                    self.adaptation_key, 
                ).to(self.device)
            elif self.cfg.adaptation_loss == "lsgan":
                discriminator = TensorDictSequential(
                    TensorDictModule(TConv(128), [("agents", "observation_h")], ["condition"]),
                    CatTensors([self.adaptation_key, "condition"], "condition", del_keys=False),
                    TensorDictModule(
                        nn.Sequential(make_mlp([256]), nn.LazyLinear(1)),
                        ["condition"], ["label"]
                    )
                )
                self.adaptation_loss = LSGAN(
                    self.adaptation_module, 
                    discriminator,
                    self.adaptation_key, 
                ).to(self.device)
            elif self.cfg.adaptation_loss == "value":
                self.adaptation_module = TensorDictModule(
                    TConv(fake_input[self.adaptation_key].shape[-1]), 
                    [("agents", "observation_h")], [self.adaptation_key]
                ).to(self.device)
                self.adaptation_module(fake_input)
                self.adaptation_loss = ValueDeviation(
                    self.encoder,
                    self.adaptation_module,
                    self.critic
                ).to(self.device)
            elif self.cfg.adaptation_loss == "action":
                self.adaptation_module = TensorDictModule(
                    TConv(fake_input[self.adaptation_key].shape[-1]), 
                    [("agents", "observation_h")], [self.adaptation_key]
                ).to(self.device)
                self.adaptation_module(fake_input)
                self.adaptation_loss = ActionDistDiv(
                    self.encoder,
                    self.adaptation_module,
                    self.actor
                ).to(self.device)
            elif self.cfg.adaptation_loss == "action_value":
                self.adaptation_module = TensorDictModule(
                    TConv(fake_input[self.adaptation_key].shape[-1]), 
                    [("agents", "observation_h")], [self.adaptation_key]
                ).to(self.device)
                self.adaptation_module(fake_input)
                self.adaptation_loss = ActionValue(
                    self.encoder,
                    self.adaptation_module,
                    self.actor,
                    self.critic
                ).to(self.device)
            else:
                raise ValueError(self.cfg.adaptation_loss)


        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=5e-4)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
    
    def forward(self, tensordict: TensorDict):
        self._get_context(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("_feature", "loc", "scale", "context", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        if self.phase == "encoder":
            info = self._train_policy(tensordict)
        elif self.phase == "adaptation":
            info = self._train_adaptation(tensordict)
        elif self.phase == "finetune":
            with hold_out_net(self.encoder):
                info = self._train_policy(tensordict.clone())
            info.update(self._train_adaptation(tensordict.clone()))
        else:
            raise RuntimeError()
        return info
    
    def _get_context(self, tensordict: TensorDict):
        assert tensordict.get("context", None) is None
        if self.phase == "encoder":
            self.encoder(tensordict)
        elif self.phase in ("adaptation", "finetune"):
            if self.adaptation_key != "context":
                self.encoder(tensordict)
            self.adaptation_module(tensordict)
        assert tensordict.get("context", None) is not None
        return tensordict

    def _train_policy(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            self._get_context(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        dones = (
            tensordict[("next", "done")]
            .expand(-1, -1, self.n_agents)
            .unsqueeze(-1)
        )
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
        # self.encoder(tensordict)
        self._get_context(tensordict)
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

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
        self.encoder_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()
        self.encoder_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])
    
    def _train_adaptation(self, tensordict: TensorDict):
        with torch.no_grad():
            tensordict = self.encoder(tensordict)
        
        info = self.adaptation_loss.update(tensordict)
        
        return {f"adaptation/{k}": v for k, v in info.items()}

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)

    def adaptation_loss_traj(self, traj: TensorDictBase):
        """
        Computes the changes of the adaptation loss in an episode. Not for training.
        """
        
        td_target = self.critic(self.encoder(traj.exclude(self.adaptation_key)))
        td_pred = self.critic(self.adaptation_module(traj.exclude(self.adaptation_key)))
        mse = F.mse_loss(
            td_target.get(self.adaptation_key),
            td_pred.get(self.adaptation_key),
            reduction="none"
        ).mean((1, 2))
        value_error = F.mse_loss(
            td_target.get("state_value"),
            td_pred.get("state_value"),
            reduction="none"
        ).mean((1, 2))
        return {"mse": mse.cpu(), "value_error": value_error.cpu()}


class MSE(nn.Module):
    def __init__(self, adaptation_module: TensorDictModule, key: str):
        super().__init__()
        self.adaptation_module = adaptation_module
        self.key = key
        self.opt = torch.optim.Adam(self.adaptation_module.parameters())
    
    def forward(self, tensordict):
        target = tensordict.get(self.key)
        pred = self.adaptation_module(tensordict).get(self.key)
        loss = F.mse_loss(pred, target)
        # return TensorDict({"loss_mse": loss}, [])
        return loss

    def update(self, tensordict):
        info = []
        for epoch in range(4):
            for batch in make_batch(tensordict, 8):
                loss = self(batch)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                info.append(loss)
        return {"adapt_loss": torch.stack(info).mean().item()}


from torchrl.objectives.utils import hold_out_net
class GAN(nn.Module):
    def __init__(
        self, 
        adaptation_module: TensorDictModule, 
        discriminator: TensorDictModule, 
        key: str
    ):
        super().__init__()
        self.adaptation_module = adaptation_module
        self.discriminator = discriminator
        self.key = key
        self.loss = F.binary_cross_entropy_with_logits
        self.opt_g = torch.optim.Adam(self.adaptation_module.parameters(), lr=5e-4)
        self.opt_d = torch.optim.Adam(self.adaptation_module.parameters(), lr=5e-4)
    
    def forward(self, tensordict):
        true_label = self.discriminator(tensordict).get("label")
        with torch.no_grad():
            self.adaptation_module(tensordict)
        fake_label = self.discriminator(tensordict).get("label")
        loss_d = (
            self.loss(true_label, torch.ones_like(true_label))
            + self.loss(fake_label, torch.zeros_like(fake_label))
        ) * 0.5
        accuracy = (
            (true_label.detach().sigmoid().round() == 1).sum()
            + (fake_label.detach().sigmoid().round() == 0).sum()
        ) / (true_label.numel() + fake_label.numel())
        with hold_out_net(self.discriminator):
            fake_label = self.adaptation_module(tensordict).get("label")
            loss_g = self.loss(fake_label, torch.ones_like(fake_label))
        # print(d_loss.item(), g_loss.item(), accuracy.item())
        return TensorDict({"loss_d": 0.5 * loss_d, "loss_g": loss_g, "accuracy": accuracy}, [])
    
    def update(self, tensordict: TensorDict):
        self.forward(tensordict)
        return 


class LSGAN(nn.Module):
    def __init__(
        self, 
        adaptation_module, 
        discriminator, 
        key
    ):
        super().__init__()
        self.adaptation_module = adaptation_module
        self.discriminator = discriminator
        self.key = key
    
    def forward(self, tensordict):
        true_label = self.discriminator(tensordict).get("label")
        self.adaptation_module(tensordict)
        fake_label = self.discriminator(tensordict).get("label")
        loss = (
            (fake_label + 1).square().mean() + 
            (true_label - 1).square().mean()
        )
        return loss




class ActionDistDiv(nn.Module):
    def __init__(
        self,
        encoder: TensorDictModule,
        adaptation_module: TensorDictModule,
        actor: ProbabilisticActor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adaptation_module = adaptation_module
        self.actor = actor
        self.opt = torch.optim.Adam(self.adaptation_module.parameters())
    
    def forward(self, tensordict: TensorDictBase):
        with torch.no_grad():
            td = self.encoder(tensordict.exclude("context"))
            target = self.actor.get_dist(td)
        with hold_out_net(self.actor):
            td = self.adaptation_module(tensordict.exclude("context"))
            pred = self.actor.get_dist(td)
        loss = D.kl_divergence(pred, target).mean()
        # loss = D.kl_divergence(target, pred).mean()
        return loss
    
    def update(self, tensordict: TensorDictBase):
        info = []
        for epoch in range(4):
            for batch in make_batch(tensordict, 8):
                loss = self(batch)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                info.append(loss)
        return {"adapt_loss": torch.stack(info).mean().item()}


class ValueDeviation(nn.Module):
    def __init__(
        self,
        encoder: TensorDictModule,
        adaptation_module: TensorDictModule,
        critic: TensorDictModule,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adaptation_module = adaptation_module
        self.critic = critic
        self.opt = torch.optim.Adam(self.adaptation_module.parameters())
    
    def forward(self, tensordict: TensorDictBase):
        with torch.no_grad():
            td = self.encoder(tensordict.exclude("context"))
            target = self.critic(td).get("state_value")
        
        with hold_out_net(self.critic):
            td = self.adaptation_module(tensordict.exclude("context"))
            pred = self.critic(td).get("state_value")
        loss = F.mse_loss(pred, target)
        return loss
    
    def update(self, tensordict: TensorDictBase):
        info = []
        for epoch in range(4):
            for batch in make_batch(tensordict, 8):
                loss = self(batch)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                info.append(loss)
        return {"adapt_loss": torch.stack(info).mean().item()}


class ActionValue(nn.Module):
    def __init__(
        self,
        encoder: TensorDictModule,
        adaptation_module: TensorDictModule,
        actor: ProbabilisticActor,
        critic: TensorDictModule
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adaptation_module = adaptation_module
        self.actor = actor
        self.critic = critic
        self.opt = torch.optim.Adam(self.adaptation_module.parameters())
    
    def forward(self, tensordict: TensorDictBase):
        with torch.no_grad():
            td = self.encoder(tensordict.exclude("context"))
            value_target = self.critic(td).get("state_value")
            action_target = self.actor.get_dist(td)
        
        with hold_out_net(self.critic), hold_out_net(self.actor):
            td = self.adaptation_module(tensordict.exclude("context"))
            value_pred = self.critic(td).get("state_value")
            action_pred = self.actor.get_dist(td)
        loss = (
            D.kl_divergence(action_pred, action_target).mean()
            + F.mse_loss(value_pred, value_target)
        )
        return loss
    
    def update(self, tensordict: TensorDictBase):
        info = []
        for epoch in range(4):
            for batch in make_batch(tensordict, 8):
                loss = self(batch)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                info.append(loss)
        return {"adapt_loss": torch.stack(info).mean().item()}


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]