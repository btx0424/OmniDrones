import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, make_functional
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec
from torchrl.modules import ProbabilisticActor, NormalParamWrapper, TanhNormal
import torch.distributions as D

from torchrl.data import CompositeSpec, TensorSpec

from torch.cuda import _device_t
from typing import List, Union, Tuple, Sequence

from tensordict.utils import NestedKey


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
    module = NormalParamWrapper(make_mlp(hidden_units))
    td_module = TensorDictModule(
        module,
        in_keys=actor_in_keys,
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=td_module,
        in_keys=actor_in_keys,
        out_keys=actor_out_keys,
        spec=action_spec,
        return_log_prob=True,
        distribution_class=D.Normal,
    )
    return actor


if __name__ == "__main__":
    actor_in_keys = ["observation"]
    actor_out_keys = ["action"]
    actor_spec = UnboundedContinuousTensorSpec(
        shape=(4, 4),
    )
    actor = make_actor(
        hidden_units=[128, 256, 128],
        actor_in_keys=actor_in_keys,
        actor_out_keys=actor_out_keys,
        action_spec=actor_spec,
    )

    td = TensorDict(
        {"observation": torch.randn(3, 4)},
        [
            3,
        ],
    )

    td = actor(td)
    print(td)
