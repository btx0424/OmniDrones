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

    from omni.isaac.orbit.scene import InteractiveSceneCfg
    from omni.isaac.orbit.assets import AssetBaseCfg
    from omni.isaac.orbit.terrains import TerrainImporterCfg
    import omni.isaac.orbit.sim as sim_utils

    class MyEnv(IsaacEnv):

        def __init__(self, cfg):
            super().__init__(cfg)
            # the `__init__` method invokes `_design_scene` to create the scene
            # after that, all the entities created are managed by `self.scene`
            print(self.scene)

            # let's get the drone entity
            # now it accounts for multiple drones
            self.drone: Multirotor = self.scene["drone"]
            self.payload_body_id = self.drone.find_bodies("payload")[0][0]
            self.payload_traj = []

            self.default_init_state = (
                self.drone.data.default_root_state
                .clone()
            )

            # set a target position for the drone
            self.target_pos = self.default_init_state[:, :3].clone()
            self.target_pos[:, 0] -= 1.0
            self.target_pos[:, 1] -= 1.0
            self.target_pos[:, 2] += 1.0

        def _design_scene(self):
            import omni.isaac.core.utils.prims as prim_utils
            import omni.physx.scripts.utils as script_utils
            import omni.isaac.core.objects as objects
            import omni_drones.utils.kit as kit_utils
            from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import UsdFileCfg
            from omni_drones.utils.orbit import _spawn_from_usd_file, clone, multi
            
            from pxr import Usd, UsdPhysics

            BAR_LENGTH = 0.8
            PAYLOAD_RADIUS = 0.04
            PAYLOAD_MASS = 0.3

            def spawn_with_payload(
                prim_path: str,
                cfg: UsdFileCfg,
                translation: tuple[float, float, float] | None = None,
                orientation: tuple[float, float, float, float] | None = None,
            ) -> Usd.Prim:
                prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
                bar = prim_utils.create_prim(
                    prim_path=prim_path + "/bar",
                    prim_type="Capsule",
                    translation=(0., 0., -BAR_LENGTH / 2.),
                    attributes={"radius": 0.01, "height": BAR_LENGTH}
                )
                UsdPhysics.RigidBodyAPI.Apply(bar)
                UsdPhysics.CollisionAPI.Apply(bar)
                massAPI = UsdPhysics.MassAPI.Apply(bar)
                massAPI.CreateMassAttr().Set(0.001)

                base_link = prim_utils.get_prim_at_path(prim_path + "/base_link")
                stage = prim_utils.get_current_stage()
                joint = script_utils.createJoint(stage, "D6", bar, base_link)
                joint.GetAttribute("limit:rotX:physics:low").Set(-120)
                joint.GetAttribute("limit:rotX:physics:high").Set(120)
                joint.GetAttribute("limit:rotY:physics:low").Set(-120)
                joint.GetAttribute("limit:rotY:physics:high").Set(120)
                UsdPhysics.DriveAPI.Apply(joint, "rotX")
                UsdPhysics.DriveAPI.Apply(joint, "rotY")
                joint.GetAttribute("drive:rotX:physics:damping").Set(2e-6)
                joint.GetAttribute("drive:rotY:physics:damping").Set(2e-6)

                payload = objects.DynamicSphere(
                    prim_path=prim_path + "/payload",
                    translation=(0., 0., -BAR_LENGTH),
                    radius=PAYLOAD_RADIUS,
                    mass=PAYLOAD_MASS
                )
                joint = script_utils.createJoint(stage, "Fixed", bar, payload.prim)
                kit_utils.set_collision_properties(
                    prim_path + "/bar", contact_offset=0.02, rest_offset=0.001
                )
                kit_utils.set_collision_properties(
                    prim_path + "/payload", contact_offset=0.02, rest_offset=0.001
                )
                return prim
            
            spawn_with_payload = clone(spawn_with_payload)
            spawn_with_payload = multi(spawn_with_payload)
                
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
        
                drone = HUMMINGBIRD_CFG
                drone.prim_path = "{ENV_REGEX_NS}/Robot_0"
                drone.spawn.func = spawn_with_payload

            return SceneCfg(num_envs=cfg.num_envs, env_spacing=2.5)

        def _reset_idx(self, env_ids: torch.Tensor):
            # since we have multiple parallel environments
            # the environment offset is added to the initial state
            init_state = self.default_init_state[env_ids]
            init_state[:, :3] += self.scene.env_origins[env_ids]
            
            self.drone.write_root_state_to_sim(init_state, env_ids)
            self.payload_traj.clear()
        
        def debug_vis(self):
            payload_pos_w = self.drone.data.body_pos_w[0, self.payload_body_id]
            self.payload_traj.append(payload_pos_w.clone())
            self.debug_draw.plot(torch.stack(self.payload_traj))
    
    env: MyEnv = MyEnv(cfg)

    def policy(tensordict: TensorDict):
        root_pos_e = (
            env.drone.data.root_pos_w
            - env.scene.env_origins
        )
        target_pos = env.target_pos - root_pos_e
        target_yaw = torch.zeros(*env.drone.shape, 1, device=env.device)
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