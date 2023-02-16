import torch
import hydra
import os
import time
from tensordict import TensorDict
from tqdm import tqdm
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.learning.collectors import SyncDataCollector

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs import Hover

    env = Hover(cfg, headless=cfg.headless)
    def policy(tensordict):
        actions: TensorDict = env.action_spec.rand()
        # for k, v in actions.items():
        #     v[:] = 0.1
        tensordict.update(actions)
        return tensordict
    
    collector = SyncDataCollector(
        env, policy, device="cuda", split_trajs=False,
        frames_per_batch=env.num_envs * 32
    )

    pbar = tqdm(collector)
    for i, data in enumerate(pbar):
        print(data)
        pbar.set_postfix(dict(rollout_fps=collector._fps))
        
    simulation_app.close()

if __name__ == "__main__":
    main()