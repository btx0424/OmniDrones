import torch
import hydra
import os
import time
from tensordict import TensorDict
from tqdm import tqdm
from functorch import vmap
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
    from omni_drones.controllers import LeePositionController

    env = Hover(cfg, headless=cfg.headless)
    controller = LeePositionController(9.81, env.drone.params).to(env.device)

    def policy(tensordict):
        state = tensordict["drone.obs"]
        controller_state = tensordict.get("controller_state", {})
        relative_state = state[..., :13].clone()
        target_pos, quat, linvel, angvel = torch.split(relative_state, [3, 4, 3, 3], dim=-1)
        control_target = torch.cat([
            target_pos, torch.zeros_like(linvel), torch.zeros_like(target_pos[..., [0]])], dim=-1)
        relative_state[..., :3] = 0.
        cmds, controller_state = vmap(vmap(controller))(relative_state, control_target, controller_state)
        tensordict["drone.action"] = cmds
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