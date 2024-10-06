import hydra
import os
import torch
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from tensordict import TensorDict


@hydra.main(config_path=os.path.dirname(__file__), config_name="demo_2")
def main(cfg):
    app = init_simulation_app(cfg)

    # due to the design of Isaac Sim, these imports are only available 
    # after the SimulationApp instance is created
    from omni_drones.envs.isaac_env import IsaacEnv
    from omni_drones.robots.assets import Multirotor, HUMMINGBIRD_CFG

    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.assets import AssetBaseCfg
    from omni.isaac.lab.terrains import TerrainImporterCfg
    import omni.isaac.lab.sim as sim_utils

    class MyEnv(IsaacEnv):

        def __init__(self, cfg):
            super().__init__(cfg)
            # the `__init__` method invokes `_design_scene` to create the scene
            # after that, all the entities created are managed by `self.scene`
            print(self.scene)

            # let's get the drone entity
            # now it accounts for multiple drones
            self.drone: Multirotor = self.scene["drone"]
            self.default_init_state = (
                self.drone.data.default_root_state
                .clone()
            )

            # set a target position for the drone
            self.target_pos = self.default_init_state[:, :3].clone()
            self.target_pos[:, 0] -= 1.0
            self.target_pos[:, 1] -= 1.0
            # self.target_pos[:, 2] += 1.0
            self.target_yaw = torch.zeros(self.drone.shape, device=self.device)
            self.target_yaw[:] = torch.pi / 2

        def _design_scene(self):
            # the scene is created from a SceneCfg object in a declarative way
            # see the docstring of `InteractiveSceneCfg` for more details
            class SceneCfg(InteractiveSceneCfg):
                terrain = TerrainImporterCfg(
                    prim_path="/World/ground",
                    terrain_type="plane",
                    collision_group=-1,
                )
                
                # lights
                light = AssetBaseCfg(
                    prim_path="/World/light",
                    spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
                )
                sky_light = AssetBaseCfg(
                    prim_path="/World/skyLight",
                    spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
                )

                drone = HUMMINGBIRD_CFG.replace(
                    prim_path="{ENV_REGEX_NS}/Robot_0", 
                )

            return SceneCfg(num_envs=cfg.num_envs, env_spacing=2.5)

        def _reset_idx(self, env_ids: torch.Tensor):
            # since we have multiple parallel environments
            # the environment offset is added to the initial state
            init_state = self.default_init_state[env_ids]
            init_state[:, :3] += self.scene.env_origins[env_ids]
            
            self.drone.write_root_state_to_sim(init_state, env_ids)
    
    env: MyEnv = MyEnv(cfg)
    env.drone.multirotor_data.drag_coef[:] = 0.5

    def policy(tensordict: TensorDict):
        root_pos_e = (
            env.drone.data.root_pos_w
            - env.scene.env_origins
        )
        target_pos = env.target_pos - root_pos_e
        target_yaw = env.target_yaw.unsqueeze(1)
        action = torch.cat([target_pos, target_yaw], dim=-1)
        tensordict["agents", "action"] = action
        return tensordict
    
    tensordict = env.reset()

    while True:
        tensordict = policy(tensordict)
        # torchrl automatically handles stepping and reset for us
        _, tensordict = env.step_and_maybe_reset(tensordict)

    
if __name__ == "__main__":
    main()