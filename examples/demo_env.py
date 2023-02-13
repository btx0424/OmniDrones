import torch
import hydra
import os
import time

from tqdm import tqdm
from torchrl.collectors import SyncDataCollector
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs import Hover

    env = Hover(cfg, headless=cfg.headless)
    def policy(tensordict):
        tensordict.update(env.action_spec.rand())
        return tensordict
    
    collector = SyncDataCollector(
        env, policy, device="cuda", split_trajs=False,
        frames_per_batch=env.num_envs * 32
    )

    for i, data in tqdm(enumerate(collector)):
        print(data)
        
    simulation_app.close()

if __name__ == "__main__":
    main()