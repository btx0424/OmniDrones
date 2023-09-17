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
from tensordict import TensorDict
from collections import defaultdict

from .mappo import MAPPOPolicy, make_dataset_naive
from .utils.gae import compute_gae

class HAPPOPolicy(MAPPOPolicy):

    def update_actor(self, batch: TensorDict, factor: torch.Tensor, agent_id: int):
        advantages = batch["advantages"]
        actor_input = batch.select(*self.actor_in_keys)
        actor_params = self.actor_params[agent_id]

        log_probs_old = batch[self.act_logps_name]
        actor_output = self.actor(actor_input, actor_params, eval_action=True)

        log_probs_new = actor_output[self.act_logps_name]
        dist_entropy = actor_output[f"{self.agent_spec.name}.action_entropy"]

        assert advantages.shape == log_probs_new.shape == dist_entropy.shape
        
        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        policy_loss = - torch.mean(factor * torch.min(surr1, surr2) * self.act_dim)
        entropy_loss = - torch.mean(dist_entropy)

        self.actor_opt.zero_grad()
        (policy_loss - entropy_loss * self.cfg.entropy_coef).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_opt.param_groups[0]["params"], self.cfg.max_grad_norm
        )
        self.actor_opt.step()

        ess = (2 * ratio.logsumexp(0) - (2 * ratio).logsumexp(0)).exp().mean() / ratio.shape[0]
        info = TensorDict({
            "policy_loss": policy_loss.item(),
            "actor_grad_norm": grad_norm.item(),
            "entropy": - entropy_loss.item(),
            "ESS": ess.item()
        }, [])
        return info, factor * ratio.detach()
    
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.select(*self.train_in_keys, strict=False)
        next_tensordict = tensordict["next"][:, -1]
        with torch.no_grad():
            value_output = self.value_op(next_tensordict)

        rewards = tensordict.get(("next", "reward", f"{self.agent_spec.name}.reward"))
        if rewards.shape[-1] != 1:
            rewards = rewards.sum(-1, keepdim=True)

        values = tensordict["state_value"]
        next_value = value_output["state_value"].squeeze(0)

        if hasattr(self, "value_normalizer"):
            values = self.value_normalizer.denormalize(values)
            next_value = self.value_normalizer.denormalize(next_value)

        dones = self._get_dones(tensordict)

        tensordict["advantages"], tensordict["returns"] = compute_gae(
            rewards,
            dones,
            values,
            next_value,
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
        )

        advantages_mean = tensordict["advantages"].mean()
        advantages_std = tensordict["advantages"].std()
        if self.normalize_advantages:
            tensordict["advantages"] = (tensordict["advantages"] - advantages_mean) / (
                advantages_std + 1e-8
            )

        if hasattr(self, "value_normalizer"):
            self.value_normalizer.update(tensordict["returns"])
            tensordict["returns"] = self.value_normalizer.normalize(
                tensordict["returns"]
            )

        train_info = []

        for ppo_epoch in range(self.ppo_epoch):
            dataset = make_dataset_naive(
                tensordict,
                int(self.cfg.num_minibatches),
                self.minibatch_seq_len if hasattr(self, "minibatch_seq_len") else 1,
            )
            for minibatch in dataset:
                factor = torch.ones(minibatch[self.act_logps_name].shape[0], 1, device=minibatch.device)
                actor_batch = minibatch.select(*self.actor_in_keys, "advantages", self.act_logps_name)
                actor_batch.batch_size = [*minibatch.shape, self.agent_spec.n]
                critic_batch = minibatch.select(*self.critic_in_keys, "returns", "state_value")
                agent_info = []
                for agent_id in torch.randperm(self.agent_spec.n):
                    info, factor = self.update_actor(actor_batch[:, agent_id], factor, agent_id.item())
                    agent_info.append(info)
                {}
                train_info.append(TensorDict({
                    **torch.stack(agent_info).apply(torch.mean, batch_size=[]),
                    **self.update_critic(critic_batch)
                }, []))
        
        train_info = {k: v.mean().item() for k, v in torch.stack(train_info).items()}
        train_info["advantages_mean"] = advantages_mean
        train_info["advantages_std"] = advantages_std
        train_info["action_norm"] = (
            tensordict[self.act_name].float().norm(dim=-1).mean()
        )
        if hasattr(self, "value_normalizer"):
            train_info["value_running_mean"] = self.value_normalizer.running_mean.mean()
        
        return {f"{self.agent_spec.name}/{k}": v for k, v in train_info.items()}