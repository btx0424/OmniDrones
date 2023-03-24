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

def sample_sub_traj(traj, seq_len):
    t = torch.randint(0, traj.shape[0] - seq_len, (1,))
    t = t + torch.arange(seq_len)
    return traj[t]


from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec, TensorSpec
from .modules.networks import MLP, ENCODERS_MAP

def make_encoder(cfg, input_spec: TensorSpec) -> nn.Module:
    if isinstance(input_spec, (BoundedTensorSpec, UnboundedContinuousTensorSpec)):
        input_dim = input_spec.shape[-1]
        encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            MLP([input_dim] + cfg.hidden_units),
        )
        encoder.output_shape = torch.Size((cfg.hidden_units[-1],))
    elif isinstance(input_spec, CompositeSpec):
        encoder_cls = ENCODERS_MAP[cfg.attn_encoder]
        encoder = encoder_cls(input_spec)
    else:
        raise NotImplementedError(input_spec)

    return encoder

