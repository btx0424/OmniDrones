import torch
import torch.nn as nn
import torchinfo

from typing import Any, Dict
from tensordict import TensorDict

from omni_drones.envs.isaac_env import AgentSpec

ALGOS = {}

class ActorCriticPolicy:

    def __init__(self, cfg,
        agent_type: AgentSpec,
        device: str="cuda",
    ) -> None:
        self.cfg = cfg
        self.agent_type = agent_type
        self.device = device

        self.actor: nn.Module
        self.critic: nn.Module
        
        self.make_model()
        self.summary()
    
    def make_model(self): ...
    
    def summary(self) -> str:
        summaries = []
        if hasattr(self, "actor"):
            summaries.append(str(torchinfo.summary(self.actor)))
        if hasattr(self, "critic"):
            summaries.append(str(torchinfo.summary(self.critic)))
        summary = "\n".join(summaries)
        return summary

    def policy_op(self, tensordict: TensorDict, training: bool):
        """
        
        """

    def value_op(self, tensordict: TensorDict, training: bool):
        """
        Compute the value of the given state.
        """

    def model_op(self, tensordict: TensorDict, training: bool):
        """
        
        """
    
    def setup(self, envs): ...

    def on_step(self): ...

    def state_dict(self) -> Dict[str, Any]: ... 

    def load_state_dict(self): ...

    def train(self):
        if hasattr(self, "actor"):
            self.actor.train()
        if hasattr(self, "critic"):
            self.critic.train()
    
    def eval(self):
        if hasattr(self, "actor"):
            self.actor.eval()
        if hasattr(self, "critic"):
            self.critic.eval()
        
    @property
    def agent_name(self):
        return self.agent_type.name
    
    @property
    def num_agents(self):
        return self.agent_type.n

    @property
    def obs_name(self):
        return self.agent_type.name + ".obs"
    
    @property
    def states_name(self):
        return self.agent_type.name + ".state"
    
    @property
    def act_name(self):
        return self.agent_type.name + ".action"

        