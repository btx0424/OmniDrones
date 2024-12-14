import hydra
import os
import torch

from omegaconf import OmegaConf
from omni_drones import init_simulation_app
from tensordict import TensorDict
from omni_drones.utils.torchrl import SyncDataCollector
from torchrl.envs.transforms import StepCounter, InitTracker, Compose, TransformedEnv

@hydra.main(config_path=os.path.dirname(__file__), config_name="demo")
def main(cfg):
    app = init_simulation_app(cfg)
    OmegaConf.set_struct(cfg, False)
    
    from omni_drones.envs.racing_simple import RacingSimple
    cfg.task.rewards = RacingSimple.DEFAULT_REWARD_CONFIG
    cfg.task.termination = RacingSimple.DEFAULT_TERMINATION_CONFIG

    env = RacingSimple(cfg)
    transform = Compose(StepCounter(), InitTracker())
    env = TransformedEnv(env, transform)
    # a dummy tensordict to show the input/output to the environment
    fake_td = env.fake_tensordict()
    print(fake_td)

    # a simple policy that takes random actions
    def policy(tensordict: TensorDict):
        tensordict.update(env.full_action_spec.rand())
        return tensordict
    
    # very similar to `DataLoader` in a common PyTorch training loop
    collector = SyncDataCollector(
        env,
        policy, # `policy` should be a callable that reads from and writes to a tensordict
        frames_per_batch=32 * env.num_envs,
        total_frames=-1,
        device=env.device,
        return_same_td=True
    )

    for i, data in enumerate(collector):
        # do something with the data
        if i ==0: print(data)
        pass

    
if __name__ == "__main__":
    main()