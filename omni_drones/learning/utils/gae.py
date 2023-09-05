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
