from torchrl.modules import IndependentNormal
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


def make_actor(
    hidden_units: Sequence[int],
    actor_in_keys: List[NestedKey],
    actor_out_keys: List[NestedKey],
    action_spec: TensorSpec,
) -> TensorDictModule:
    assert isinstance(action_spec, UnboundedContinuousTensorSpec)

    module = MyNormalParamWrapper(make_mlp(hidden_units), action_spec.shape[-1])
    td_module = TensorDictModule(
        module,
        in_keys=actor_in_keys,
        out_keys=["loc", "scale"],
    )
    actor = MyProbabilisticActor(
        module=td_module,
        in_keys=["loc", "scale"],
        out_keys=actor_out_keys,
        spec=action_spec,
        return_log_prob=True,
        log_prob_key=("agent", "action_logp"),
        distribution_class=IndependentNormal,
    )
    return actor
