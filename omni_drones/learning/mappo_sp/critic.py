import torch
import torch.nn as nn
import torch.distributions as D

from torch.cuda import _device_t
from typing import List, Union, Tuple, Sequence
from tensordict.utils import NestedKey
from tensordict.nn import TensorDictModule

from torchrl.data import TensorSpec
from torchrl.data import UnboundedContinuousTensorSpec
from .utils import MyNormalParamWrapper, MyProbabilisticActor, make_mlp


def make_critic(
    hidden_units: List[int],
    critic_in_keys: List[NestedKey],
    critic_out_keys: List[NestedKey],
):
    return TensorDictModule(
        nn.Sequential(make_mlp(hidden_units), nn.LazyLinear(1)),
        critic_in_keys,
        critic_out_keys,
    )
