import torch
import torch.nn as nn


def soft_update(target: nn.Module, source: nn.Module, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update(target: nn.Module, source: nn.Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

