import torch

num_envs = 128
device = "cpu"
env_ids = torch.arange(num_envs // 2)

ball_turn = torch.randint(low=0, high=4, size=(num_envs,))

obs = torch.randn(num_envs, 4, 5)

a = obs[torch.arange(len(env_ids)), ball_turn, :]
print(a.shape)