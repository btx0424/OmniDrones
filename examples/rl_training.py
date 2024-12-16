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
    cfg.task.max_episode_length = 1000
    cfg.task.rewards = RacingSimple.DEFAULT_REWARD_CONFIG
    cfg.task.termination = RacingSimple.DEFAULT_TERMINATION_CONFIG

    env = RacingSimple(cfg)
    transform = Compose(
        StepCounter(),
        InitTracker(),
        VecNorm([("agents", "observation")])
    )
    env = TransformedEnv(env, transform)

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
        pprint.pprint(info)

    
if __name__ == "__main__":
    main()