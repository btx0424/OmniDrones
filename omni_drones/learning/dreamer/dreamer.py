import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from omni_drones.learning.common import MyBuffer
from omni_drones.learning.modules.distributions import MultiOneHotCategorical
from collections import defaultdict

class TwoHot(D.distribution):
    ...


class SymlogTransform(D.Transform):
    ...


def rollout(module: TensorDictModule, tensordict: TensorDict):


    return 
    
class DreamerPolicy():
    def __init__(
        self,
        cfg,
        agent_spec,
    ) -> None:
        self.cfg = cfg
        self.buffer = MyBuffer()

        self.obs_pred_dist = defaultdict()
        self.value_pred_dist = TwoHot
        self.reward_pred_dist = ...
        
        self.wm = ...
        self.task_actor = ...
        self.critic = ...
        self.model_opt = torch.optim.Adam(self.wm.parameters())
        self.actor_opt = torch.optim.Adam(self.task_actor)
        self.critic_opt = torch.optim.Adam(self.critic)

        self.observe = TensorDictSequential(

        )
        self.imagine = TensorDictSequential(

        )

    def __call__(self, tensordict: TensorDict):
        ...
    
    def train_op(self, tensordict: TensorDict):
        self.buffer.extend(tensordict)

        metrics = defaultdict(list)

        for _ in range(self.cfg.gradient_steps):
            batch = self.buffer.sample()

            get_dist = lambda logits: MultiOneHotCategorical(logits=logits)
            batch = rollout(self.observe, batch)
            prior_logits = batch["prior_logits"]
            post_logits = batch["post_logits"]
            reward_pred = batch["reward_pred"]
            discount_pred = batch["discount_pred"]
            model_losses = {}
            model_losses["dyn"] = D.kl_divergence(get_dist(post_logits.detach()), get_dist(prior_logits))
            model_losses["rep"] = D.kl_divergence(get_dist(post_logits), get_dist(prior_logits.detach()))
            model_losses["reward"] = self.reward_pred_dist(reward_pred).log_prob()
            model_losses["discount"] = F.binary_cross_entropy_with_logits(discount_pred, )
            model_loss = sum(model_losses.values())
            self.model_opt.zero_grad()
            model_loss.backward()
            wm_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.wm.parameters())
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



