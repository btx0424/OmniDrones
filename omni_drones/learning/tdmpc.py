import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from functorch import vmap

from tensordict import TensorDict
from torchrl.objectives.utils import hold_out_net

from .common import MyBuffer, make_encoder, soft_update
from .modules.networks import MLP
import copy
from collections import defaultdict
from tqdm import tqdm

def orthogonal_init(m):
	"""Orthogonal layer initialization."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	elif isinstance(m, nn.Conv2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if m.bias is not None:
			nn.init.zeros_(m.bias)

class LinearSchedule:
    def __init__(self, start, finish, steps):
        self.start = start
        self.finish = finish
        self.delta = (finish - start) / steps
    
    def __call__(self, step):
        if self.start > self.finish:
            return max(self.start + step * self.delta, self.finish)
        else:
            return min(self.start + step * self.delta, self.finish)
    
    @classmethod
    def make(cls, schedule: str):
        import re
        match = re.match(r'linear\((.+),(.+),(.+)\)', schedule)
        start, finish, steps = [float(g) for g in match.groups()]
        return cls(start, finish, steps)

        

class TOLD(nn.Module):
    def __init__(
        self, 
        cfg,
        observation_spec,
        action_spec,
    ):
        super().__init__()
        action_dim = action_spec.shape[-1]
        encoder = make_encoder(cfg.encoder, observation_spec)
        self.encoder = nn.Sequential(
            encoder,
            nn.ELU(),
            nn.Linear(encoder.output_shape.numel(), cfg.hidden_dim)
        )
        self.dynamics = MLP(
            [cfg.hidden_dim + action_dim, *cfg.dynamics.hidden_units, cfg.hidden_dim], 
            nn.LayerNorm
        )
        self.reward = MLP(
            [cfg.hidden_dim + action_dim, 1]
        )
        self.q1 = MLP([cfg.hidden_dim + action_dim, 1], nn.LayerNorm)
        self.q2 = MLP([cfg.hidden_dim + action_dim, 1], nn.LayerNorm)
        self.apply(orthogonal_init)
    
    def h(self, obs):
        return self.encoder(obs)
    
    def next(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.dynamics(x), self.reward(x)
    
    def q(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return torch.cat([self.q1(x), self.q2(x)], dim=-1)


def mse_loss(input, target, weights):
    return torch.mean(F.mse_loss(input, target, reduction="none") * weights)


class TDMPCPolicy:
    def __init__(
        self,
        cfg,
        agent_spec,
        device
    ):
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        self.obs_name = f"{self.agent_spec.name}.obs"
        self.action_name = ("action", f"{self.agent_spec.name}.action")
        self.reward_name = ("next", "reward", f"{self.agent_spec.name}.reward")

        self.model = TOLD(
            cfg.model, 
            self.agent_spec.observation_spec, 
            self.agent_spec.action_spec
        ).to(self.device)
        self.model_target = copy.deepcopy(self.model)
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.model.lr)
        
        self.action_dim = self.agent_spec.action_spec.shape[-1]
        self.actor = MLP(
            [self.cfg.model.hidden_dim, *self.cfg.actor.hidden_units, self.action_dim],
            nn.LayerNorm
        ).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor.lr)

        self.gradient_steps = int(self.cfg.gradient_steps)
        self.buffer = MyBuffer(self.cfg.buffer_size, device=self.device)
        self.horizon_schedule = LinearSchedule.make(self.cfg.horizon_schedule)
        self.std_schedule = LinearSchedule.make(self.cfg.std_schedule)

        # planning
        self.num_samples = self.cfg.num_samples
        self.num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        self.cem_iterations = int(self.cfg.cem_iterations)
        self.cem_num_elites = int(self.cfg.cem_num_elites)
        self.cem_momentum = self.cfg.cem_momentum
        self.cem_temperature = self.cfg.cem_temperature

        self._step = 0
        init_horizon = int(self.horizon_schedule(0))
        self._prev_mean = (
            self.agent_spec.action_spec
            .zero()
            .expand(-1, init_horizon, -1)
        )

    def __call__(self, tensordict: TensorDict):
        obs = tensordict[self.obs_name]
        is_init = tensordict.get("is_init").unsqueeze(-1)
        prev_mean = self._prev_mean * (1 - is_init.float())
        action, mean, std, elite_value = vmap(self.plan, randomness="different")(
            obs.squeeze(1), prev_mean, step=self._step
        )
        self._prev_mean = mean
        self._plan_std = std
        self._plan_elite_value = elite_value
        tensordict.set(self.action_name, action)
        return tensordict
    
    def pi(self, z, std):
        action = torch.tanh(self.actor(z))
        action_noise = action.clone().normal_(0, std)
        action = torch.clamp(action + action_noise, -1, 1)
        return action

    def estimate_value(self, z, actions):
        G = 0
        discount = 1
        for t in range(actions.shape[0]):
            z, r = self.model.next(z, actions[t])
            G += discount * r
            discount *= self.cfg.gamma
        qs = self.model.q(z, self.pi(z, self.cfg.min_std))
        G += discount * qs.min(-1, keepdims=True).values
        return G
        
    @torch.no_grad()
    def plan(self, obs, prev_mean, step, eval_mode=False):

        horizon = int(self.horizon_schedule(step))
        z0 = self.model.h(obs)
        z = z0.repeat(self.num_pi_trajs, 1)
        pi_actions = []
        for t in range(horizon):
            pi_action = self.pi(z, self.cfg.min_std)
            pi_actions.append(pi_action)
            z, _ = self.model.next(z, pi_action)
        pi_actions = torch.stack(pi_actions)

        mean = torch.cat([
            prev_mean[1:],
            torch.zeros(horizon-(prev_mean.shape[0]-1), self.action_dim, device=self.device)
        ])
        std = torch.ones_like(mean)
        std_lb = self.std_schedule(step)
        
        z = z0.repeat(self.num_samples+self.num_pi_trajs, 1)
        for i in range(self.cem_iterations):
            actions = torch.clamp(
                mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(horizon, self.num_samples, self.action_dim, device=self.device)
                -1, 1
            )
            actions = torch.cat([actions, pi_actions], dim=1)
        
            value = self.estimate_value(z, actions)
            elite_idxs = torch.topk(value.squeeze(1), self.cem_num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cem_temperature*(elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
            _std = torch.clamp(_std, std_lb, 2)
            mean, std = self.cem_momentum * mean + (1 - self.cem_momentum) * _mean, _std

        action_idx = D.Categorical(probs=score.squeeze(-1)).sample()
        action = elite_actions[0][action_idx.unsqueeze(0)]
        if not eval_mode:
            action += std[0] * torch.randn_like(action)
        return action, mean, std, elite_value

    def train_op(self, data: TensorDict):
        self.buffer.extend(data)

        if len(self.buffer) < self.cfg.buffer_size:
            print(f"filling buffer: {len(self.buffer)}/{self.cfg.buffer_size}")
            return {}
        
        self._step += data[("next", "done")].sum().item()
        metrics = defaultdict(list)

        for step in tqdm(range(self.cfg.gradient_steps)):
            batch = self.buffer.sample(self.cfg.batch_size, self.cfg.batch_length)

            obs = batch[self.obs_name]
            next_obs = batch[("next", self.obs_name)]
            action = batch[self.action_name]
            reward = batch[self.reward_name]
            not_done = 1. - batch[("next", "done")].float()

            # Representation
            z = self.model.h(obs[:, 0])
            zs = []
            rs = []
            qs = []
            for t in range(self.cfg.horizon):
                qs.append(self.model.q(z, action[:, t]))
                z, r = self.model.next(z, action[:, t])
                zs.append(z)
                rs.append(r)
            
            zs = torch.stack(zs, 1)
            rs = torch.stack(rs, 1)
            qs = torch.stack(qs, 1)
            with torch.no_grad():
                target_zs = self.model_target.h(next_obs)
                new_action = self.pi(self.model.h(next_obs), self.cfg.min_std)
                qs_target = self.model_target.q(target_zs, new_action)
                td_target = reward + self.cfg.gamma * qs_target.min(-1, keepdims=True).values
            
            weights = torch.cumprod(not_done * self.cfg.rho, dim=1).unsqueeze(-1) / self.cfg.rho
            consistency_loss = mse_loss(zs, target_zs, weights)
            reward_loss = mse_loss(rs, reward, weights)
            value_loss = sum(
                mse_loss(q.unsqueeze(-1), td_target, weights) 
                for q in qs.unbind(-1)
            )
            
            total_loss = (
                self.cfg.consistency_coef * consistency_loss
                + self.cfg.reward_coef * reward_loss
                + self.cfg.value_coef * value_loss
            ) 
            self.model_opt.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.model_opt.step()
            
            with hold_out_net(self.model):
                zs = self.model.h(obs)
                qs = self.model.q(zs, self.pi(zs, self.cfg.min_std))
                actor_loss = - torch.mean(qs.min(-1, keepdims=True).values * weights)
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            if step % self.cfg.target_update_interval == 0:
                soft_update(self.model_target, self.model, self.cfg.tau)

            metrics["cosistency_loss"].append(consistency_loss)
            metrics["reward_loss"].append(reward_loss)
            metrics["value_loss"].append(value_loss)
            metrics["model_grad_norm"].append(grad_norm)
            metrics["actor_loss"].append(actor_loss)

        metrics = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        metrics["horizon"] = self.horizon_schedule(self._step)
        metrics["std"] = self.std_schedule(self._step)
        metrics["plan_std"] = self._plan_std.mean().item()
        metrics["plan_elite_value"] = self._plan_elite_value.mean().item()
        return metrics

