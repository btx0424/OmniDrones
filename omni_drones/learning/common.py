import torch
import torch.nn as nn
import torchinfo

from typing import Optional, Sequence, Union, Type
from tensordict import TensorDict

from omni_drones.envs.isaac_env import AgentSpec

ALGOS = {}

class ActorCriticPolicy:

    def __init__(self, cfg,
        agent_spec: AgentSpec,
        act_name: str = None,
        device: str="cuda",
    ) -> None:
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        self.actor: nn.Module
        self.critic: nn.Module
        
        if act_name is not None:
            self._act_name = act_name

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
        
    @property
    def agent_name(self):
        return self.agent_spec.name
    
    @property
    def num_agents(self):
        return self.agent_spec.n

    @property
    def obs_name(self):
        return self.agent_spec.name + ".obs"
    
    @property
    def states_name(self):
        return self.agent_spec.name + ".state"
    
    @property
    def act_name(self):
        if hasattr(self, "_act_name"):
            return self._act_name
        else:
            return self.agent_spec.name + ".action"
