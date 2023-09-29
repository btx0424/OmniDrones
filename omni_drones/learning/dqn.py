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
from tensordict.nn import TensorDictModule

from torchrl.data import (
    DiscreteTensorSpec,
    TensorDictReplayBuffer
)
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler

import copy
from tqdm import tqdm
from tensordict import TensorDict
from .common import soft_update, make_encoder

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DuelingQNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        V = self.value_head(self.feature_layer(state))
        A = self.advantage_head(self.feature_layer(state))
        Qs = V + A - A.mean(dim=-1, keepdim=True)
        return Qs

class EpsilonGreedyActionSelector(nn.Module):
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_finish: float = 0.05,
        anneal_time: float = 50000,
        decay="linear",
    ):
        super().__init__()
        assert decay in ["linear", "exp"]
        self.epsilon_start = epsilon_start
        self.epsilon_finish = epsilon_finish
        self.anneal_time = anneal_time
        self.decay = decay
        self.delta = (epsilon_start - epsilon_finish) / anneal_time

    def forward(self, agent_qs: torch.Tensor, t: float):
        if self.decay == "linear":
            epsilon = max(self.epsilon_start - self.delta * t, self.epsilon_finish)
        elif self.decay == "exp":
            raise NotImplementedError

        num_actions = agent_qs.shape[-1]
        random_actions = torch.randint(
            0, num_actions, agent_qs.shape[:-1], device=agent_qs.device
        ).unsqueeze(-1)
        greedy_actions = agent_qs.argmax(dim=-1, keepdim=True)
        assert random_actions.shape == greedy_actions.shape
        actions = torch.where(
            torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon,
            random_actions,
            greedy_actions,
        )
        return actions, epsilon
    
class DQNPolicy:

    def __init__(self,
        cfg,
        agent_spec,
        device: str="cuda",
    ) -> None:
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device        
        self.batch_size = int(cfg.batch_size)

        self.obs_name = f"{self.agent_spec.name}.obs"
        self.act_name = ("action", f"{self.agent_spec.name}.action")
        self.reward_name = f"{self.agent_spec.name}.reward"

        if not isinstance(agent_spec.action_spec, DiscreteTensorSpec):
            raise ValueError("Only discrete action spaces are supported for DQN.")
        
        self.num_actions = agent_spec.action_spec.space.n

        self.gamma = cfg.gamma
        
        self.target_update_interval = cfg.target_update_interval
        self.target_update_count = 0

        # self.use_dueling = cfg.use_dueling
        self.use_double = cfg.use_double
        
        encoder = make_encoder(cfg.q_net, agent_spec.observation_spec)
        self.q_net = TensorDictModule(
            nn.Sequential(
                encoder,
                nn.Linear(encoder.output_shape.numel(), self.num_actions),
            ),
            in_keys=[self.obs_name],
            out_keys=["q_net_output"],
        ).to(self.device)
        self.action_selector = EpsilonGreedyActionSelector(
            self.cfg.epsilon_start,
            self.cfg.epsilon_finish,
            self.cfg.anneal_time,
        )
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.cfg.q_net.lr)
        
        self.replay_buffer = TensorDictReplayBuffer(
            batch_size=self.batch_size,
            storage=LazyTensorStorage(max_size=self.cfg.buffer_size, device=self.device),
            sampler=RandomSampler()
        )
        self.t = 0
        self.epsilon = 0

    def __call__(self, tensordict: TensorDict, deterministic: bool=False) -> TensorDict:

        self.q_net(tensordict)
        action, self.epsilon = self.action_selector(tensordict["q_net_output"], t=self.t)
        tensordict[self.act_name] = action

        self.t += 1
        return tensordict

    def train_op(self, data: TensorDict):
        self.replay_buffer.extend(data.reshape(-1))

        if len(self.replay_buffer) < self.cfg.buffer_size:
            print(f"{len(self.replay_buffer)} < {self.cfg.buffer_size}")
            return {}
        
        infos = []

        with tqdm(range(1, int(self.cfg.gradient_steps)+1)) as t:
            for gradient_step in t:
                transition = self.replay_buffer.sample().to(self.device)

                action_taken = transition[self.act_name] # [batchsize, drone, action_dim=1]
                reward  = transition[("next", "reward", f"{self.agent_spec.name}.reward")] # [batchsize, drone, 1]
                next_dones  = transition[("next", "done")].float().unsqueeze(-1) # [batchsize, drone, 1]
        
                # get Q_target
                with torch.no_grad():
                    if self.use_double:
                        action = self.q_net(transition["next"])["q_net_output"].argmax(dim=-1)
                    else:
                        action = self.target_q_net(transition["next"])["q_net_output"].argmax(dim=-1)
                    
                    next_qs = self.target_q_net(transition["next"])["q_net_output"]
                    next_q = next_qs.gather(dim=-1, index=action.unsqueeze(-1)) # [batchsize, drone, 1
                    target_q = (reward + self.gamma * next_q * (1 - next_dones)).detach() # [batch_size, drone, 1]

                q = self.q_net(transition["next"])["q_net_output"].gather(dim=-1, index=action_taken) # [batch_size, drone, 1]
                loss = F.mse_loss(q, target_q)
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                if gradient_step % self.target_update_interval:
                    with torch.no_grad():
                        soft_update(self.target_q_net, self.q_net, self.cfg.tau)
                
                infos.append(TensorDict({
                    "q_loss": loss,
                    "q_taken_mean": q.mean(),
                    "q_taken_max": q.max(),
                    "grad_norm": grad_norm,
                }, []))
                t.set_postfix({"q_loss": loss.item()})
        
        infos = torch.stack(infos)
        infos = {k: torch.mean(v).item() for k, v in infos.items()}
        infos["epsilon"] = self.epsilon

        return infos

