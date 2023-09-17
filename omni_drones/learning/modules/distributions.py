# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import functools
import math
from numbers import Number
from typing import List, Tuple, Union

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

D.Distribution.set_default_validate_args(False)


def init(module: nn.Module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def expln(x):
    """A smooth, continuous positive mapping presented in "State-Dependent Exploration for Policy Gradient Methods".

    https://people.idsia.ch/~juergen/ecml2008rueckstiess.pdf

    """
    out = torch.empty_like(x)
    idx_neg = x <= 0
    out[idx_neg] = x[idx_neg].exp()
    out[~idx_neg] = x[~idx_neg].log1p() + 1
    return out


_mappings = {
    "softplus": F.softplus,
    "exp": torch.exp,
    "relu": torch.relu,
    "expln": expln,
}


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=False, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.log_std = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_std = torch.broadcast_to(torch.exp(self.log_std), action_mean.shape)
        dist = D.Independent(D.Normal(action_mean, action_std), 1)
        return dist


# class SafeTanhTransform(D.TanhTransform):
#     """Safe version of TanhTransform that avoids NaNs."""

#     def _inverse(self, y: torch.Tensor):
#         eps = torch.finfo(y.dtype).eps
#         y = y.clamp(-1 + eps, 1 - eps)
#         return torch.atanh(y)


# class TanhIndependentNormal(D.TransformedDistribution):
#     arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

#     def __init__(
#         self,
#         loc: torch.Tensor,
#         scale: torch.Tensor,
#         min: Union[torch.Tensor, Number] = -1.0,
#         max: Union[torch.Tensor, Number] = 1.0,
#         event_dims=1,
#     ):
#         self.min = torch.as_tensor(min, device=loc.device).broadcast_to(loc.shape)
#         self.max = torch.as_tensor(max, device=loc.device).broadcast_to(loc.shape)
#         self._loc = (self.min + self.max) / 2
#         self._scale = (self.max - self.min) / 2
#         self.eps = torch.finfo(loc.dtype).eps
#         base_dist = D.Independent(D.Normal(loc, scale), event_dims)
#         t = SafeTanhTransform()

#         super().__init__(base_dist, t)

#     def sample(self, sample_shape: torch.Size = torch.Size()):
#         return super().sample(sample_shape) * self._scale + self._loc

#     def rsample(self, sample_shape: torch.Size = torch.Size()):
#         return super().rsample(sample_shape) * self._scale + self._loc

#     def log_prob(self, value: torch.Tensor):
#         return super().log_prob(
#             ((value - self._loc) / self._scale).clamp(-1 + self.eps, 1 - self.eps)
#         )

#     @property
#     def mode(self):
#         m = self.base_dist.mode
#         return torch.tanh(m)

#     @property
#     def mean(self):
#         return self.mode

#     def entropy(self):
#         return -self.log_prob(self.rsample(self.batch_shape))
from torchrl.modules.distributions import TanhNormal

class IndependentNormal(D.Independent):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = D.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)


class IndependentBeta(D.Independent):
    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        min: Union[float, torch.Tensor] = 0.0,
        max: Union[float, torch.Tensor] = 1.0,
        event_dims: int = 1,
    ):
        self.min = torch.as_tensor(min, device=alpha.device).broadcast_to(alpha.shape)
        self.max = torch.as_tensor(max, device=alpha.device).broadcast_to(alpha.shape)
        self.scale = self.max - self.min
        self.eps = torch.finfo(alpha.dtype).eps
        base_dist = D.Beta(alpha, beta)
        super().__init__(base_dist, event_dims)

    def sample(self, sample_shape: torch.Size = torch.Size()):
        return super().sample(sample_shape) * self.scale + self.min

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        return super().rsample(sample_shape) * self.scale + self.min

    def log_prob(self, value: torch.Tensor):
        return super().log_prob(
            ((value - self.min) / self.scale).clamp(self.eps, 1.0 - self.eps)
        )


class IndependentNormalModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dependent_std: bool = False,
        scale_mapping: str = "exp",
        scale_lb: float = 1e-4,
    ):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.operator = nn.Linear(input_dim, output_dim * 2)
        else:
            self.operator = nn.Linear(input_dim, output_dim)
            self.log_std = nn.Parameter(torch.zeros(output_dim))
        if isinstance(scale_mapping, str):
            self.scale_mapping = _mappings[scale_mapping]
        elif callable(self.scale_mapping):
            self.scale_mapping = scale_mapping
        else:
            raise ValueError("scale_mapping must be a string or a callable function.")
        self.scale_lb = scale_lb

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor]:
        if self.state_dependent_std:
            loc, scale = self.operator(tensor).chunk(2, -1)
            scale = self.scale_mapping(scale).clamp_min(self.scale_lb)
        else:
            loc, scale = self.operator(tensor), self.scale_mapping(
                self.log_std
            ).clamp_min(self.scale_lb)
        return IndependentNormal(loc, scale)


class TanhNormalWithEntropy(TanhNormal):

    def entropy(self):
        return -self.log_prob(self.sample())


class TanhIndependentNormalModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scale_mapping: str = "softplus",
        state_dependent_std: bool = True,
        scale_lb: float = 1e-4,
        min: Union[torch.Tensor, Number] = -1.0,
        max: Union[torch.Tensor, Number] = 1.0,
        event_dims=1,
    ):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.operator = nn.Linear(input_dim, output_dim * 2)
        else:
            self.operator = nn.Linear(input_dim, output_dim)
            self.log_std = nn.Parameter(torch.zeros(output_dim))
        if isinstance(scale_mapping, str):
            self.scale_mapping = _mappings[scale_mapping]
        elif callable(self.scale_mapping):
            self.scale_mapping = scale_mapping
        else:
            raise ValueError("scale_mapping must be a string or a callable function.")
        self.scale_lb = scale_lb
        self.dist_cls = functools.partial(
            TanhNormalWithEntropy, min=min, max=max, event_dims=event_dims
        )

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor]:
        if self.state_dependent_std:
            loc, scale = self.operator(tensor).chunk(2, -1)
            scale = self.scale_mapping(scale).clamp_min(self.scale_lb)
        else:
            loc = self.operator(tensor)
            scale = self.scale_mapping(self.log_std).clamp_min(self.scale_lb)
        return self.dist_cls(loc, scale)


class IndependentBetaModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scale_mapping: str = "softplus",
        min: Union[float, torch.Tensor] = 0.0,
        max: Union[float, torch.Tensor] = 1.0,
        event_dims: int = 1,
    ):
        super().__init__()
        self.operator = nn.Linear(input_dim, output_dim * 2)
        if isinstance(scale_mapping, str):
            self.scale_mapping = _mappings[scale_mapping]
        elif callable(self.scale_mapping):
            self.scale_mapping = scale_mapping
        else:
            raise ValueError("scale_mapping must be a string or a callable function.")
        self.dist_cls = functools.partial(
            IndependentBeta, min=min, max=max, event_dims=event_dims
        )

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor]:
        alpha, beta = self.operator(tensor).chunk(2, -1)
        alpha = self.scale_mapping(alpha)
        beta = self.scale_mapping(beta)
        return self.dist_cls(alpha, beta)


class MultiCategorical(D.Distribution):
    def __init__(
        self,
        logits: List[torch.Tensor] = None,
        probs: List[torch.Tensor] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

        if probs is not None:
            self.base_dists = [D.Categorical(probs=probs_) for probs_ in probs]
        else:
            self.base_dists = [D.Categorical(logits=logits_) for logits_ in logits]

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        return torch.stack(
            [dist.sample(sample_shape) for dist in self.base_dists], dim=-1
        )

    @lazy_property
    def logits(self):
        return [dist.logits for dist in self.base_dists]

    @lazy_property
    def probs(self):
        return [dist.probs for dist in self.base_dists]

    @property
    def mode(self):
        return torch.stack([dist.mode for dist in self.base_dists], dim=-1)

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        return torch.stack(
            [dist.sample(sample_shape) for dist in self.base_dists], dim=-1
        )

    def log_prob(self, value):
        return torch.stack(
            [
                dist.log_prob(v)
                for dist, v in zip(self.base_dists, value.unbind(dim=-1))
            ],
            dim=-1,
        ).sum(dim=-1)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self.base_dists], dim=-1).sum(
            dim=-1
        )


class MultiCategoricalModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dims: Union[List[int], torch.Tensor],
    ):
        super().__init__()
        self.operator = nn.Linear(input_dim, sum(output_dims))
        self.output_dims = (
            output_dims.tolist()
            if isinstance(output_dims, torch.Tensor)
            else output_dims
        )

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor]:
        logits = self.operator(tensor)
        logits = logits.split(self.output_dims, dim=-1)
        return MultiCategorical(logits=logits)


class MultiOneHotCategorical(D.Independent):
    def __init__(
        self,
        logits: torch.Tensor = None,
        probs: torch.Tensor = None,
        unimix: float = 0.01
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        
        if logits is not None:
            probs = F.softmax(logits, dim=-1)
        
        probs = probs * (1. - unimix) + unimix / probs.shape[-1]
        super().__init__(
            D.OneHotCategoricalStraightThrough(probs=probs),
            1
        )

class TwoHot(D.Distribution):
    def __init__(
        self, 
        logits: torch.Tensor, 
        low=-20.0, 
        high=20.0, 
    ):
        super().__init__(batch_shape=logits.shape[:-1])
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=logits.shape[-1]).to(logits.device)

    @property
    def mean(self):
        return torch.sum(self.probs * self.buckets, -1, keepdim=True)

    @property
    def mode(self):
        return self.mean

    def log_prob(self, x):
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

