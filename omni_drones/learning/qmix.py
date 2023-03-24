import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from functorch import vmap
from torchrl.data import (
    TensorSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    CompositeSpec,
)
from .modules.rnn import GRU
from .modules.networks import ENCODERS_MAP, MLP
from .common import MyBuffer, soft_update, make_encoder
import copy
from tqdm import tqdm

class QMIX:
    def __init__(
        self,
        cfg,
        agent_spec,
        device=None,
    ):
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device
        self.agent_name = agent_spec.name

        n_agents = agent_spec.n
        num_actions = agent_spec.action_spec.shape[0]
        obs_name = f"{self.agent_name}.obs"
        state_name = f"{self.agent_name}.state"

        obs_encoder = make_encoder(cfg.q_net, agent_spec.observation_spec)
        action_selector = EpsilonGreedyActionSelector()
        hidden_dim = cfg.q_net.hidden_dim
        self.agent_q = TensorDictSequential(
            TensorDictModule(obs_encoder, [obs_name], ["hidden"]),
            TensorDictModule(
                GRU(obs_encoder.output_shape.numel(), hidden_dim),
                ["hidden"],
                ["hidden", f"{self.agent_name}.rnn_state"],
            ),
            TensorDictModule(
                nn.Linear(hidden_dim, num_actions), ["hidden"], [f"{self.agent_name}.q"]
            ),
            TensorDictModule(
                action_selector, [f"{self.agent_name}.q"], [f"{self.agent_name}.action"]
            )
        )
        self.target_agent_q = copy.deepcopy(self.agent_q)

        state_encoder = make_encoder(cfg.q_mixer, agent_spec.state_spec)
        hidden_dim = cfg.q_mixer.hidden_dim
        mixer = QMIXer(n_agents, state_encoder.output_shape.numel(), hidden_dim)
        self.mixer = TensorDictSequential(
            TensorDictModule(state_encoder, [state_name], ["mixer_hidden"]),
            TensorDictModule(mixer, [f"chosen_q", "mixer_hidden"], [f"q_tot"]),
        )
        self.target_mixer = copy.deepcopy(self.mixer)

        params = list(self.agent_q.parameters()) + list(self.mixer.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg.lr)

        self.rb = MyBuffer(cfg.buffer_size, device=self.device)

    def __call__(self, tensordict: TensorDict):
        self.agent_q(tensordict)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        reward = tensordict[("next", "reward", f"{self.agent_name}.reward")]
        if reward.dim() == 4: # [N, L, M, *]
            # force shared reward
            tensordict[("next", "reward", f"{self.agent_name}.reward")] = reward.sum(dims=[-1, -2])

        self.rb.extend(tensordict)

        infos = []
        t = tqdm(range(self.cfg.gradient_steps))
        for gradient_step in t:
            batch: TensorDict = self.rb.sample(self.cfg.batch_size)
            chosen_actions = batch[f"{self.agent_name}.action"]  # [N, L, M, 1]
            reward = batch[("next", "reward", f"{self.agent_name}.reward")]
            next_done = batch[("next", "done")].float()

            qs = self.agent_q(batch)[f"{self.agent_name}.q"]  # [N, L, M, |A|]
            chosen_action_qs = torch.gather(qs, -1, chosen_actions)

            with torch.no_grad():
                target_qs: torch.Tensor = self.target_agent_q(batch["next"])[f"{self.agent_name}.q"]
                target_max_qs = target_qs.max(dim=-1).values

            chosen_action_q_tot = self.mixer(
                batch.set("chosen_q", chosen_action_qs),
            )
            target_action_q_tot = self.target_mixer(
                batch["next"].set("chosen_q", target_max_qs),
            )

            loss = F.mse_loss(
                chosen_action_q_tot,
                reward + self.cfg.gamma * target_action_q_tot * (1 - next_done),
            )
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            infos.append(TensorDict({
                "q_loss": loss.item(),
            }))

            t.set_postfix({"q_loss": loss.item()})

            if gradient_step % self.cfg.target_update_interval == 0:
                soft_update(self.agent_q, self.target_agent_q, self.cfg.tau)
                soft_update(self.mixer, self.target_mixer, self.cfg.tau)


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

    def forward(self, agent_qs: torch.Tensor, t):
        if self.decay == "linear":
            epsilon = max(self.epsilon_finish, self.epsilon_start - self.delta * t)
        elif self.decay == "exp":
            raise NotImplementedError

        num_actions = agent_qs.shape[-1]
        random_actions = torch.randint(
            0, num_actions, agent_qs.shape[:-1], device=agent_qs.device
        )
        greedy_actions = agent_qs.argmax(dim=-1)
        actions = torch.where(
            torch.rand_like(agent_qs.shape[:-1]) < epsilon,
            random_actions,
            greedy_actions,
        )
        return actions


class QMIXer(nn.Module):
    def __init__(
        self,
        n_agents: int,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(self.input_dim, self.hidden_dim * self.n_agents)
        self.w2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.b1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.b2 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, agent_qs, state):
        assert agent_qs.shape[0] == state.shape[0]
        assert agent_qs.shape[-1] == self.n_agents
        w1 = torch.abs(self.w1(state))
        b1 = self.b1(state)
        w2 = torch.abs(self.w2(state))
        b2 = self.b2(state)

        h = F.elu(F.linear(agent_qs, w1, b1))
        q_tot = F.linear(h, w2, b2)
        return q_tot
