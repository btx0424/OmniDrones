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
            nn.Linear(encoder.output_shape.numel(), cfg.hidden_dim)
        )
        self.action_proj = nn.Linear(action_dim, cfg.hidden_dim)
        self.dynamics = nn.Sequential(
            MLP([cfg.hidden_dim * 2, *cfg.dynamics.hidden_units], nn.LayerNorm),
            nn.Linear(cfg.dynamics.hidden_units[-1], cfg.hidden_dim)
        )
        self.reward = nn.Sequential(
            MLP([cfg.hidden_dim * 2, cfg.hidden_dim], nn.LayerNorm),
            nn.Linear(cfg.hidden_dim, 1)
        )
        self.cont =  nn.Sequential(
            MLP([cfg.hidden_dim * 2, cfg.hidden_dim], nn.LayerNorm),
            nn.Linear(cfg.hidden_dim, 1)
        )
        q_units = [cfg.hidden_dim * 2, cfg.hidden_dim]
        self.q1 = nn.Sequential(MLP(q_units, nn.LayerNorm), nn.Linear(q_units[-1], 1))
        self.q2 = nn.Sequential(MLP(q_units, nn.LayerNorm), nn.Linear(q_units[-1], 1))
        self.apply(orthogonal_init)
    
    def h(self, obs):
        return self.encoder(obs)
    
    def next(self, z, a):
        a = self.action_proj(a)
        x = torch.cat([z, a], dim=-1)
        return self.dynamics(x), self.reward(x), self.cont(x)
    
    def q(self, z, a):
        a = self.action_proj(a)
        x = torch.cat([z, a], dim=-1)
        return torch.cat([self.q1(x), self.q2(x)], dim=-1)


def mse_loss(input, target, weights):
    return torch.mean(F.mse_loss(input, target, reduction="none") * weights)

__REDUCE__ = lambda b: 'mean' if b else 'none'

def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))

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
        self.actor = nn.Sequential(
            MLP([self.cfg.model.hidden_dim, *self.cfg.actor.hidden_units], nn.LayerNorm),
            nn.Linear(self.cfg.actor.hidden_units[-1], self.action_dim)
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
        action, mean, std, elite_value, pi_value = vmap(self.plan, randomness="different")(
            obs.squeeze(1), prev_mean, step=self._step
        )
        self._prev_mean = mean
        self._plan_std = std
        self._plan_elite_value = elite_value
        self._pi_value = pi_value
        tensordict.set(self.action_name, action)
        return tensordict
    
    def pi(self, z, std):
        action = torch.tanh(self.actor(z))
        action_noise = (
            action
            .clone()
            .normal_(0, std)
            .clip(-0.2, 0.2)
        )
        action = torch.clamp(action + action_noise, -1, 1)
        return action

    def estimate_value(self, z, actions):
        G = 0
        discount = 1
        for t in range(actions.shape[0]):
            z, r, c = self.model.next(z, actions[t])
            G += discount * r
            discount *= self.cfg.gamma * c.round()
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
            z, _, _ = self.model.next(z, pi_action)
        pi_actions = torch.stack(pi_actions)

        mean = torch.cat([
            prev_mean[1:],
            torch.zeros(horizon-(prev_mean.shape[0]-1), self.action_dim, device=self.device)
        ])
        std = 2 * torch.ones_like(mean)
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
            action = torch.clamp(action + std[0] * torch.randn_like(action), -1, 1)
        pi_value = value[-self.num_pi_trajs:]
        return action, mean, std, elite_value, pi_value

    def train_op(self, data: TensorDict):
        self.buffer.extend(data)

        if len(self.buffer) < self.cfg.buffer_size:
            print(f"filling buffer: {len(self.buffer)}/{self.cfg.buffer_size}")
            return {}
        
        self._step += data[("next", "done")].sum().item()
        metrics = defaultdict(list)

        for step in tqdm(range(self.cfg.gradient_steps)) if self.cfg.verbose else range(self.cfg.gradient_steps):
            batch = self.buffer.sample(self.cfg.batch_size, self.cfg.horizon)

            obs = batch[self.obs_name].squeeze(1)
            next_obs = batch[("next", self.obs_name)].squeeze(1)
            action = batch[self.action_name].squeeze(1)
            reward = batch[self.reward_name].squeeze(1)
            not_done = (1. - batch[("next", "done")].float()).unsqueeze(-1)

            # Representation
            z = self.model.h(obs[:, 0])
            zs = [z.detach()]
            consistency_loss, reward_loss, cont_loss, value_loss = 0, 0, 0, 0
            rho = not_done[:, 0]
            for t in range(self.cfg.horizon):
                qs = self.model.q(z, action[:, t])
                z, r, c = self.model.next(z, action[:, t])
                with torch.no_grad():
                    next_z = self.model_target.h(next_obs[:, t])
                    next_action = self.pi(next_z, self.cfg.min_std)
                    next_qs = self.model_target.q(next_z, next_action)
                    td_target = reward[:, t] + self.cfg.gamma * not_done[:, t] * next_qs.min(-1, keepdims=True).values
                zs.append(z.detach())

                # Losses
                consistency_loss += rho * torch.mean(mse(z, next_z), dim=-1, keepdim=True)
                reward_loss += rho * mse(r, reward[:, t])
                cont_loss += rho * mse(c, not_done[:, t])
                value_loss += rho * (mse(qs[..., 0:1], td_target) + mse(qs[..., 1:2], td_target))
                rho = rho * self.cfg.rho * not_done[:, t]
            
            total_loss = (
                self.cfg.consistency_coef * consistency_loss
                + self.cfg.reward_coef * reward_loss
                + self.cfg.reward_coef * cont_loss
                + self.cfg.value_coef * value_loss
            ).mean()

            self.model_opt.zero_grad()
            total_loss.backward()
            model_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.model_opt.step()
            
            if step % self.cfg.actor_delay == 0:
                actor_loss = 0
                with hold_out_net(self.model):
                    # for t, z in enumerate(zs):
                    #     # a = self.pi(z, self.cfg.min_std)
                    #     a = self.pi(z, 0)
                    #     q = self.model.q(z, a).min(-1, keepdims=True).values
                    #     actor_loss += -q.mean() * (self.cfg.rho ** t)
                    zs = torch.stack(zs)
                    a = self.pi(zs, self.cfg.min_std)
                    q = self.model.q(zs, a).min(-1, keepdims=True).values
                    rho = torch.cumprod(torch.ones_like(q) * self.cfg.rho, 0) / self.cfg.rho
                    actor_loss = -(q * rho).mean()
                self.actor_opt.zero_grad()
                actor_loss.backward()
                actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()
                metrics["actor_loss"].append(actor_loss)
                metrics["actor_grad_norm"].append(actor_grad_norm)

            if step % self.cfg.target_update_interval == 0:
                soft_update(self.model_target, self.model, self.cfg.tau)

            metrics["model_loss/cosistency"].append(consistency_loss)
            metrics["model_loss/reward"].append(reward_loss)
            metrics["model_loss/cont"].append(cont_loss)
            metrics["model_grad_norm"].append(model_grad_norm)
            metrics["value_loss"].append(value_loss)

        metrics = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        metrics["horizon"] = self.horizon_schedule(self._step)
        metrics["std"] = self.std_schedule(self._step)
        metrics["plan_std"] = self._plan_std.mean().item()
        metrics["plan_elite_value"] = self._plan_elite_value.mean().item()
        metrics["plan_pi_value"] = self._pi_value.mean().item()
        return metrics

    def state_dict(self):
        state_dict = {
            "actor": self.actor.state_dict(),
            "model": self.model.state_dict(),
            "step": self._step
        }
        return state_dict