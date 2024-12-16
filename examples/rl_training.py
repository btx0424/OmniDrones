import hydra
import os
import torch

from omegaconf import OmegaConf
from omni_drones import init_simulation_app
from tensordict import TensorDict
from torchrl.envs.transforms import StepCounter, InitTracker, Compose, VecNorm, TransformedEnv

from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.learning.ppo.ppo import PPOPolicy, PPOConfig
import pprint

@hydra.main(config_path=f"{os.path.dirname(__file__)}", config_name="demo")
def main(cfg):
    app = init_simulation_app(cfg)
    OmegaConf.set_struct(cfg, False)
    
    from omni_drones.envs.racing_simple import RacingSimple
    from omni_drones.envs.single.track import Track
    task_cls = RacingSimple

    cfg.task.max_episode_length = 1000
    cfg.task.rewards = task_cls.DEFAULT_REWARD_CONFIG
    cfg.task.termination = task_cls.DEFAULT_TERMINATION_CONFIG
    env = task_cls(cfg)

    transform = Compose(
        StepCounter(),
        InitTracker(),
        VecNorm([("agents", "observation")])
    )
    env = TransformedEnv(env, transform)

    stats = {k: torch.tensor(0., device=env.device) for k in env.reward_spec["stats"].keys()}
    count = torch.tensor(0., device=env.device)
    decay = 0.99

    policy = PPOPolicy(
        PPOConfig(),
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=env.device
    )
    
    # very similar to `DataLoader` in a common PyTorch training loop
    # `policy` should be a callable that reads from and writes to a tensordict
    collector = SyncDataCollector(
        env,
        policy=policy.get_rollout_policy("train"), 
        frames_per_batch=policy.cfg.train_every * env.num_envs,
        total_frames=-1,
        device=env.device,
        return_same_td=True
    )
    
    for i, data in enumerate(collector):
        # do something with the data
        info = policy.train_op(data)
        
        done = data["next", "done"].squeeze(-1)
        if done.any():
            data = data[done]
            for key in stats.keys():
                stats[key].mul_(decay).add_(data["next", "stats", key].sum())
            count.mul_(decay).add_(done.sum())

        info["stats"] = {k: (v / count).item() for k, v in stats.items()}
        pprint.pprint(info)
    
if __name__ == "__main__":
    main()