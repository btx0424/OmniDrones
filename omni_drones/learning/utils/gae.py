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


def compute_gae(
    reward: torch.Tensor,  # [N, T, k]
    done: torch.Tensor,  # [N, T, 1]
    value: torch.Tensor,  # [N, T, k]
    next_value: torch.Tensor,  # [N, k]
    gamma=0.99,
    lmbda=0.95,
):
    assert reward.shape == value.shape

    not_done = 1.0 - done.float()
    batch_size, num_steps = not_done.shape[:2]
    gae = 0
    advantages = torch.zeros_like(reward)
    for step in reversed(range(num_steps)):
        delta = (
            reward[:, step] 
            + gamma * next_value * not_done[:, step] 
            - value[:, step]
        )
        advantages[:, step] = gae = delta + (gamma * lmbda * not_done[:, step] * gae)
        next_value = value[:, step]

    returns = advantages + value  # aka. value targets
    return advantages, returns


def compute_gae_(
    reward: torch.Tensor,  # [T, N, k]
    done: torch.Tensor,  # [T, N, 1]
    value: torch.Tensor,  # [T, N, k]
    next_value: torch.Tensor,  # [N, k]
    gamma=0.99,
    lmbda=0.95,
):
    assert reward.shape == value.shape

    not_done = 1.0 - done.float()
    num_steps = not_done.shape[0]
    gae = 0
    advantages = torch.zeros_like(reward)
    for step in reversed(range(num_steps)):
        delta = reward[step] + gamma * next_value * not_done[step] - value[step]
        advantages[step] = gae = delta + (gamma * lmbda * not_done[step] * gae)
        next_value = value[step]

    returns = advantages + value  # aka. value targets
    return advantages, returns
