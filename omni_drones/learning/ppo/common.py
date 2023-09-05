import torch
import torch.nn as nn


class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step] 
                + self.gamma * next_value[:, step] * not_done[:, step] 
                - value[:, step]
            )
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae)
        returns = advantages + value
        return advantages, returns