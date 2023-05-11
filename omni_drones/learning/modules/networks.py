from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from torch import Tensor
from torchrl.data import CompositeSpec, TensorSpec


def register(_map: Dict, name=None):
    def decorator(func):
        _name = name or func.__name__
        assert _name not in _map
        _map[_name] = func
        return func

    return decorator


class MLP(nn.Module):
    def __init__(
        self,
        num_units: Sequence[int],
        normalization: Union[str, nn.Module] = None,
        activation_class: nn.Module = nn.ELU,
        activation_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        layers = []
        if activation_kwargs is not None:
            activation_class = partial(activation_class, **activation_kwargs)
        if isinstance(normalization, str):
            normalization = getattr(nn, normalization, None)
        for i, (in_dim, out_dim) in enumerate(zip(num_units[:-1], num_units[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(num_units) - 1:
                layers.append(activation_class())
            if normalization is not None:
                layers.append(normalization(out_dim))
        self.layers = nn.Sequential(*layers)
        self.input_dim = num_units[0]
        self.output_shape = torch.Size((num_units[-1],))

    def forward(self, x: torch.Tensor):
        return self.layers(x)


def split(x, split_shapes, split_sizes):
    return [
        xi.unflatten(-1, shape)
        for xi, shape in zip(torch.split(x, split_sizes, dim=-1), split_shapes)
    ]


def ij(a: torch.Tensor):
    ai = a.unsqueeze(-2).expand(*a.shape[:-2], a.shape[-2], a.shape[-2], a.shape[-1])
    aj = ai.transpose(-2, -3)
    aij = torch.cat([ai, aj], dim=-1)
    return aij


class LFF(nn.Module):
    """Learnable Fourier Features.

    Ideally should help with learning the high-frequency parts for coordinate-like inputs.

    https://openreview.net/forum?id=uTqvj8i3xv

    """

    def __init__(
        self,
        input_size,
        sigma: float = 0.01,
        fourier_dim=256,
        embed_dim=72,
        cat_input=True,
    ) -> None:
        super().__init__()
        b_shape = (input_size, fourier_dim)
        self.cat_input = cat_input
        self.B = nn.Parameter(
            torch.normal(torch.zeros(b_shape), torch.full(b_shape, sigma))
        )
        if self.cat_input:
            self.linear = nn.Linear(fourier_dim * 2 + input_size, embed_dim)
        else:
            self.linear = nn.Linear(fourier_dim * 2, embed_dim)

    def forward(self, x: Tensor):
        proj = torch.matmul(x, self.B) * (2 * torch.pi)
        if self.cat_input:
            ff = torch.cat([torch.sin(proj), torch.cos(proj), x], dim=-1)
        else:
            ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.linear(ff)


class SplitEmbedding(nn.Module):
    def __init__(
        self,
        input_spec: CompositeSpec,
        embed_dim: int = 72,
        layer_norm=True,
        embed_type="linear",
    ) -> None:
        super().__init__()
        if any(isinstance(spec, CompositeSpec) for spec in input_spec.values()):
            raise ValueError("Nesting is not supported.")
        self.input_spec = input_spec
        self.embed_dim = embed_dim
        self.num_entities = sum(spec.shape[-2] for spec in self.input_spec.values())

        if embed_type == "linear":
            self.embed = nn.ModuleDict(
                {
                    key: nn.Linear(value.shape[-1], self.embed_dim)
                    for key, value in self.input_spec.items()
                }
            )
        else:
            raise NotImplementedError(embed_type)

        if layer_norm:
            # self.layer_norm = nn.LayerNorm(embed_dim)
            self.layer_norm = nn.LayerNorm(
                (self.num_entities, embed_dim)
            )  # somehow faster

    def forward(self, tensordict: TensorDict):
        embeddings = torch.cat(
            [self.embed[key](tensordict[key]) for key in self.input_spec.keys()], dim=-2
        )
        if hasattr(self, "layer_norm"):
            embeddings = self.layer_norm(embeddings)
        return embeddings


################################## Attn Encoders ##################################
ENCODERS_MAP = {}


@register(ENCODERS_MAP)
class RelationEncoder(nn.Module):
    """
    f(sum_ij g(a_i, a_j))
    """

    def __init__(
        self,
        input_spec: CompositeSpec,
        *,
        embed_dim: int = 72,
        embed_type: str = "linear",
        layer_norm=True,
        f_units=(256, 128),
    ) -> None:
        super().__init__()
        self.output_shape = torch.Size((f_units[-1],))
        self.split_embed = SplitEmbedding(input_spec, embed_dim, layer_norm, embed_type)
        if layer_norm:
            self.g = nn.Sequential(
                MLP([embed_dim * 2, f_units[0]]), nn.LayerNorm(f_units[0])
            )
        else:
            self.g = MLP([embed_dim * 2, f_units[0]])
        self.f = MLP(f_units)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        a: torch.Tensor = self.split_embed(x)
        aij = ij(a)
        g_aij = self.g(aij)
        if mask is not None:
            if mask.shape == x.shape[:-1]:
                mask = mask.unsqueeze(-1)
            elif not mask.dim() == x.dim():
                raise RuntimeError(mask.shape)
            g_aij *= ij(mask).all(-1)
        return self.f(torch.sum(g_aij, dim=(-3, -2)))


@register(ENCODERS_MAP)
class PartialRelationEncoder(nn.Module):
    """
    f(sum_j g(a_i, a_j)), i=0, j!=i
    """

    def __init__(
        self,
        input_spec: CompositeSpec,
        *,
        embed_dim: int = 72,
        embed_type: str = "linear",
        layer_norm=True,
        f_units=(256, 128),
    ) -> None:
        super().__init__()
        self.output_shape = torch.Size((f_units[-1],))
        self.split_embed = SplitEmbedding(input_spec, embed_dim, layer_norm, embed_type)
        if layer_norm:
            self.g = nn.Sequential(
                MLP([embed_dim * 2, f_units[0]]), nn.LayerNorm(f_units[0])
            )
        else:
            self.g = MLP([embed_dim * 2, f_units[0]])
        self.f = MLP(f_units)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        a: torch.Tensor = self.split_embed(x)
        ai, aj = a.split([1, a.shape[-2] - 1], dim=-2)
        aij = torch.cat([ai.broadcast_to(aj.shape), aj], dim=-1)
        g_aij = self.g(aij)
        if mask is not None:
            if mask.shape == x.shape[:-1]:
                mask = mask.unsqueeze(-1)
            elif not mask.dim() == x.dim():
                raise RuntimeError(mask.shape)
            g_aij *= mask[..., 1:, :]
        return self.f(torch.sum(g_aij, dim=-2))


@register(ENCODERS_MAP)
class PartialAttentionEncoder(nn.Module):
    def __init__(
        self,
        input_spec: CompositeSpec,
        *,
        query_index=0,
        embed_dim: int = 128,
        embed_type: str = "linear",
        num_heads: int = 1,
        dim_feedforward=128,
        layer_norm=True,
        norm_first=False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.output_shape = torch.Size((self.embed_dim,))
        self.split_embed = SplitEmbedding(input_spec, embed_dim, layer_norm, embed_type)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        if query_index is None:
            self.query_index = ...
        elif isinstance(query_index, int):
            self.query_index = [query_index]
        else:
            self.query_index = list(query_index)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.activation = F.gelu
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None):
        """
        Args:
            x: (batch, N, dim)
            padding_mask: (batch, N)
        """
        x = self.split_embed(x)
        if self.norm_first:
            x = x[:, self.query_index] + self._pa_block(self.norm1(x), key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x[:, self.query_index] + self._pa_block(x, key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x.mean(-2)

    def _pa_block(self, x: Tensor, key_padding_mask: Optional[Tensor] = None):
        x = self.attn(
            x[:, self.query_index],
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return x

    def _ff_block(self, x: Tensor):
        x = self.linear2(self.activation(self.linear1(x)))
        return x
    

################################## Vision Encoders ##################################
def get_output_shape(net, input_size):
    _x = torch.zeros(input_size).unsqueeze(0)
    _out = net(_x)
    return _out.shape

class MixedEncoder(nn.Module):
    def __init__(
        self,
        cfg,
        vision_obs_names: List[str] = [],
        vision_encoder: nn.Module = None,
        state_encoder: Optional[nn.Module] = None,
        combine_mode: str = "concat",
    ):
        super().__init__()
        self.vision_obs_names = vision_obs_names
        self.vision_encoder = vision_encoder
        self.state_encoder = state_encoder
        self.combine_mode = combine_mode

        input_dim = self.vision_encoder.output_shape.numel()
        if self.state_encoder is not None:
            input_dim += self.state_encoder.output_shape.numel()

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            MLP(
                num_units=[input_dim] + cfg.hidden_units, 
                normalization=nn.LayerNorm if cfg.get("layer_norm", False) else None
            ),
        )

        self.output_shape = torch.Size((cfg.hidden_units[-1],))

    def forward(self, x: TensorDict):
        feats = [
            self.vision_encoder(x[obs_name]) for obs_name in x.keys() 
            if obs_name in self.vision_obs_names
        ]
        if self.state_encoder is not None:
            # TODO: attn_encoder with TensorDict input
            feats += [
                self.state_encoder(x[obs_name]) for obs_name in x.keys() 
                if obs_name not in self.vision_obs_names
            ]
        if self.combine_mode == "concat":
            feats = torch.cat(feats, dim=-1)
        else:
            raise NotImplementedError
        return self.mlp(feats)


VISION_ENCODER_MAP = {}

@register(VISION_ENCODER_MAP)
class MobileNetV3Small(nn.Module):
    def __init__(
        self,
        input_size: torch.Size,
    ) -> None:
        super().__init__()
        self.input_size = input_size[2:]
        assert self.input_size[0] in [1, 3]

        self._setup_backbone()
        self.output_shape = get_output_shape(self.forward, self.input_size)

    def _backbone_fn(self, x):
        x = self._backbone_transform(x)
        x = self._backbone(x)
        return x

    def _setup_backbone(self):
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        _transform = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms

        full_backbone = mobilenet_v3_small(weights="IMAGENET1K_V1")
        for module in full_backbone.modules():
            if isinstance(module, nn.modules.batchnorm.BatchNorm2d):
                module.track_running_stats = False

        self._backbone = nn.Sequential(
            full_backbone.features,
            full_backbone.avgpool
        )
        self._backbone_transform = _transform()

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self._backbone_fn(x)
        x = x.flatten(1)
        return x
    

@register(VISION_ENCODER_MAP)
class MobilNetV3Large(MobileNetV3Small):
    def _setup_backbone(self):
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        _transform = MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms

        full_backbone = mobilenet_v3_large(weights="IMAGENET1K_V1")
        for module in full_backbone.modules():
            if isinstance(module, nn.modules.batchnorm.BatchNorm2d):
                module.track_running_stats = False

        self._backbone = nn.Sequential(
            full_backbone.features,
            full_backbone.avgpool
        )
        self._backbone_transform = _transform()
        