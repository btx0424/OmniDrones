from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey
from torchrl.data.tensor_specs import TensorSpec
import torch

import torch.nn as nn
from numbers import Number
from typing import Sequence, Tuple, Optional, Union, List
from tensordict import TensorDictBase
from torchrl.modules.utils import mappings
from torchrl.modules import ProbabilisticActor


def make_mlp(num_units: Sequence[int]):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.ELU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


class MyNormalParamWrapper(nn.Module):
    def __init__(
        self,
        operator: nn.Module,
        output_dim: int,
        scale_mapping: str = "biased_softplus_1.0",
        scale_lb: Number = 1e-4,
    ) -> None:
        super().__init__()
        self.operator = operator
        self.scale_mapping = scale_mapping
        self.scale_lb = scale_lb
        self._linear = nn.LazyLinear(output_dim)
        self._scale = nn.Parameter(torch.zeros(output_dim))

    def forward(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor]:
        net_output = self.operator(*tensors)
        others = ()
        if not isinstance(net_output, torch.Tensor):
            net_output, *others = net_output
        loc = self._linear(net_output)
        scale = (
            mappings(self.scale_mapping)(self._scale)
            .expand_as(loc)
            .clamp_min(self.scale_lb)
        )
        return (loc, scale, *others)


class MyProbabilisticActor(ProbabilisticActor):
    def __init__(
        self,
        module: TensorDictModule,
        in_keys: Union[NestedKey, Sequence[NestedKey]],
        out_keys: Union[Sequence[NestedKey], None] = None,
        *,
        log_prob_key: Optional[NestedKey] = None,
        spec: Union[TensorSpec, None] = None,
        **kwargs,
    ):
        super().__init__(
            module, in_keys, out_keys, log_prob_key=log_prob_key, spec=spec, **kwargs
        )
        self.wanted_keys = self.in_keys + self.out_keys

        self.wanted_keys = [k for k in self.wanted_keys if k not in module.out_keys]

    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: Optional[TensorDictBase] = None,
        **kwargs,
    ):
        td: TensorDictBase = super().forward(tensordict, tensordict_out, **kwargs)
        return td.select(*self.wanted_keys)
