import torch

import torch.nn as nn
from numbers import Number
from typing import Tuple

from torchrl.modules.utils import mappings


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
