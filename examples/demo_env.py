import os
import time

import hydra
import torch
from functorch import vmap
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from tensordict import TensorDict
from tqdm import tqdm


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs import Hover
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg

    env = Hover(cfg, headless=cfg.headless)
    controller = env.drone.DEFAULT_CONTROLLER(env.drone.dt, 9.81, env.drone.params).to(
        env.device
    )

    def policy(tensordict: TensorDict):
        state = tensordict["drone.obs"]
        controller_state = tensordict.get(
            "controller_state", TensorDict({}, state.shape[:2])
        )
        relative_state = state[..., :13].clone()
        target_pos, quat, linvel, angvel = torch.split(
            relative_state, [3, 4, 3, 3], dim=-1
        )
        control_target = torch.cat(
            [
                target_pos,
                torch.zeros_like(linvel),
                torch.zeros_like(target_pos[..., [0]]),
            ],
            dim=-1,
        )
        relative_state[..., :3] = 0.0
        cmds, controller_state = vmap(vmap(controller))(
            relative_state, control_target, controller_state
        )
        tensordict[("action", "drone.action")] = cmds
        tensordict["controller_state"] = controller_state
        return tensordict

    env.enable_render = True
    env.rollout(
        max_steps=env.num_envs * 1000,
        policy=policy,
        break_when_any_done=False,
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
