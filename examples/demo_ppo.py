import torch
import hydra
import os
import time

from tqdm import tqdm
from omegaconf import OmegaConf
from setproctitle import setproctitle
from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.learning.collectors import SyncDataCollector
from omni_drones.utils.wandb import init_wandb

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs import Hover
    from omni_drones.learning.mappo import MAPPOPolicy

    env = Hover(cfg, headless=cfg.headless)
    policy = MAPPOPolicy(cfg.algo, env.agent_spec["drone"], device="cuda")
    
    collector = SyncDataCollector(
        env, policy, device="cuda", split_trajs=False,
        frames_per_batch=env.num_envs * 8
    )

    episode_stats = []
    def record_and_log_stats(done: torch.Tensor):
        done = done.squeeze(-1)
        episode_stats.append(env._tensordict[done].select("drone.return", "progress"))
        if sum(map(len, episode_stats)) >= 4096:
            print()
            stats = {}
            for k, v in torch.cat(episode_stats).items():
                v = torch.mean(v).item()
                stats[k] = v
                print(f"train/{k}: {v}")
            run.log(stats)
            episode_stats.clear()
    collector.on_reset(record_and_log_stats)
    
    pbar = tqdm(collector)
    for i, data in enumerate(pbar):
        policy.train_op(data.clone())
        pbar.set_postfix({
            "rollout_fps": collector._fps,
            "frames": collector._frames
        })
        
    simulation_app.close()

if __name__ == "__main__":
    main()