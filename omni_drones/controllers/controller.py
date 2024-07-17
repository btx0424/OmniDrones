import abc
import torch
import torch.nn as nn
from torchrl.data import TensorSpec
from typing import Dict

class ControllerBase(nn.Module):

    action_spec: TensorSpec
    REGISTRY: Dict[str, "ControllerBase"] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if cls.__name__ in ControllerBase.REGISTRY:
            raise ValueError("")
        super().__init_subclass__(**kwargs)
        ControllerBase.REGISTRY[cls.__name__] = cls
        ControllerBase.REGISTRY[cls.__name__.lower()] = cls

    @abc.abstractmethod
    def compute(self, *args, **kwargs) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def process_rl_actions(self, actions) -> torch.Tensor:
        ...
