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


import torch
import torch.nn as nn


def soft_update(target: nn.Module, source: nn.Module, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update(target: nn.Module, source: nn.Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tensordict import TensorDict
from functorch import vmap

class MyBuffer:

    def __init__(
        self, 
        max_size: int=1000, 
        device: torch.device=None
    ):
        self.storage = LazyTensorStorage(max_size=max_size, device=device)
        self._cursor = 0
    
    def extend(self, data: TensorDict):
        t = data.shape[-1]
        cursor = (self._cursor + torch.arange(t)) % self.storage.max_size
        index = self.storage.set(cursor, data.permute(1, 0))
        self._cursor = cursor[-1].item() + 1
        return index
    
    def sample(self, batch_size, seq_len: int=80):
        if seq_len > self.storage._len:
            raise ValueError(
                f"seq_len {seq_len} is larger than the current buffer length {self.storage._len}"
            )
        if isinstance(batch_size, int):
            batch_size = torch.Size([batch_size])
        else:
            batch_size = torch.Size(batch_size)
        num_samples = batch_size.numel()
        sub_sample_idx = torch.randint(0, self.storage._storage.shape[1], (num_samples,))
        sub_trajs = vmap(sample_sub_traj, in_dims=1, randomness="different")(
            self.storage[:self.storage._len, sub_sample_idx], seq_len=seq_len
        )
        return sub_trajs

    def __len__(self):
        return self.storage._len

def sample_sub_traj(traj, seq_len):
    t = torch.randint(0, traj.shape[0] - seq_len, (1,))
    t = t + torch.arange(seq_len)
    return traj[t]


from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec, TensorSpec
from .modules.networks import MLP, ENCODERS_MAP, VISION_ENCODER_MAP, MixedEncoder
from functools import partial

def make_encoder(cfg, input_spec: TensorSpec) -> nn.Module:
    if isinstance(input_spec, (BoundedTensorSpec, UnboundedContinuousTensorSpec)):
        input_dim = input_spec.shape[-1]
        encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            MLP(
                num_units=[input_dim] + cfg.hidden_units, 
                normalization=nn.LayerNorm if cfg.get("layer_norm", False) else None
            ),
        )
        encoder.output_shape = torch.Size((cfg.hidden_units[-1],))
        if cfg.get("init", None) is not None:
            init = getattr(nn.init, cfg.init.type)
            init = partial(init, **cfg.init.get("kwargs", {}))
            encoder.apply(lambda m: init_linear(m, init))
    elif isinstance(input_spec, CompositeSpec): # FIXME: add logic for composite spec with visual input and other inputs
        state_spec_dict = {}
        vision_spec_dict = {}
        for spec_name in input_spec.keys():
            if input_spec[spec_name].ndim < 5:
                state_spec_dict[spec_name] = input_spec[spec_name]
            elif input_spec[spec_name].ndim == 5:
                vision_spec_dict[spec_name] = input_spec[spec_name]
            else:
                raise ValueError
            
        # create state encoder
        if len(state_spec_dict) > 0:
            encoder_cls = ENCODERS_MAP[cfg.attn_encoder]
            state_encoder = encoder_cls(CompositeSpec(state_spec_dict))
        else:
            state_encoder = None
            print("No state encoder requried.")

        # create vision encoder
        if len(vision_spec_dict) == 0:
            assert state_encoder is not None
            encoder = state_encoder
        elif len(vision_spec_dict) == 1:
            vision_encoder_cls = VISION_ENCODER_MAP[cfg.vision_encoder]
            vision_shape = list(vision_spec_dict.values())[0].shape
            vision_encoder = vision_encoder_cls(vision_shape)
            encoder = MixedEncoder(
                cfg,
                vision_obs_names=vision_spec_dict.keys(),
                vision_encoder=vision_encoder,
                state_encoder=state_encoder
            )
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError("Multiple visual inputs are not supported for now (cuz this author is lazy)")
    else:
        raise NotImplementedError(input_spec)
        
    return encoder

@torch.no_grad()
def init_linear(module: nn.Module, weight_init):
    if isinstance(module, nn.Linear):
        weight_init(module.weight)

