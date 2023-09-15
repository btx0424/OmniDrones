from torchrl.modules import ProbabilisticActor, IndependentNormal, NormalParamWrapper
import torch
import torch.nn as nn
import torch.distributions as D

from torch.cuda import _device_t
from typing import List, Union, Tuple, Sequence
from tensordict.utils import NestedKey
from tensordict.nn import TensorDictModule

from torchrl.data import CompositeSpec, TensorSpec

from .utils import MyNormalParamWrapper

def make_mlp(num_units: Sequence[int]):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.ELU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


def make_actor(
    hidden_units: Sequence[int],
    actor_in_keys: List[NestedKey],
    actor_out_keys: List[NestedKey],
    action_spec: TensorSpec,
):
    print(action_spec.shape)
    module = MyNormalParamWrapper(make_mlp(hidden_units),action_spec.shape[-1])
    td_module = TensorDictModule(
        module,
        in_keys=actor_in_keys,
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=td_module,
        in_keys=["loc", "scale"],
        out_keys=actor_out_keys,
        spec=action_spec,
        return_log_prob=True,
        distribution_class=IndependentNormal,
    )
    return actor

