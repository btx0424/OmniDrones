import torch

from tensordict import TensorDict
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils.math import quat_from_angle_axis, quat_rotate_inverse
import omni.isaac.lab.sim as sim_utils

from omni_drones.envs import IsaacEnv, mdp
from omni_drones.robots.assets import Multirotor, FIREFLY_CFG


class RacingSimple(IsaacEnv):

    DEFAULT_REWARD_CONFIG = {
        "survival": {"weight": 1.0},
        "progress": {"weight": 1.0},
    }

    DEFAULT_TERMINATION_CONFIG = {
        "height_exceeds_range": {"range": (0.4, 4.0)},
    }

    def __init__(self, cfg):
        super().__init__(cfg)
        # the `__init__` method invokes `_design_scene` to create the scene
        # after that, all the entities created are managed by `self.scene`
        print(self.scene)

        # let's get the drone entity
        self.drone: Multirotor = self.scene["drone"]
        self.default_init_state = self.drone.data.default_root_state.clone()

        self.gate: RigidObject = self.scene["gate"]
        root_pose = self.gate.data.root_state_w.clone()
        root_pose[:, 2] += 1.5 + torch.rand(len(root_pose), device=self.device)
        theta = torch.linspace(0, torch.pi * 2, 7)[:-1]
        root_pose[:, 0] = 3 * torch.cos(theta)
        root_pose[:, 1] = 3 * torch.sin(theta)
        root_pose[:, 3:7] = quat_from_angle_axis(theta + torch.pi/2, torch.tensor([0., 0., 1.]).expand(len(theta), 3))
        self.gate.write_root_state_to_sim(root_pose)
        
        self.gate_idx = torch.zeros(self.num_envs, dtype=int, device=self.device)
        self.curr_gate_distance = torch.zeros(self.num_envs, 1, device=self.device)
        self.resolve_specs()
    
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
            
            drone = FIREFLY_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot",
            )

            gate = RigidObjectCfg(
                prim_path=[f"/World/gate_{i}" for i in range(6)],
                spawn=sim_utils.UsdFileCfg(
                    usd_path="/home/btx0424/lab/gate.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                    )
                ),
            )
        return SceneCfg(num_envs=self.cfg.num_envs, env_spacing=2.5)

    def _reset_idx(self, env_ids: torch.Tensor):
        init_state = self.default_init_state[env_ids]
        gate_idx = torch.randint(0, 6, (len(env_ids),), device=self.device)
        curr_gate_pos_w = self.gate.data.root_pos_w[gate_idx]
        next_gate_pos_w = self.gate.data.root_pos_w[(gate_idx + 1) % 6]
        init_state[:, :3] = (curr_gate_pos_w + next_gate_pos_w) / 2

        self.drone.write_root_state_to_sim(init_state, env_ids)
        self.gate_idx[env_ids] = gate_idx
    
    def _update_common(self):
        drone_pos_w = self.drone.data.root_pos_w
        drone_quat_w = self.drone.data.root_quat_w
        self.curr_gate_pos_w = self.gate.data.root_pos_w[self.gate_idx]
        self.next_gate_pos_w = self.gate.data.root_pos_w[(self.gate_idx + 1) % 6]
        self.curr_gate_pos_b = quat_rotate_inverse(drone_quat_w, self.curr_gate_pos_w - drone_pos_w)
        self.next_gate_pos_b = quat_rotate_inverse(drone_quat_w, self.next_gate_pos_w - drone_pos_w)
        
        curr_gate_distance = self.curr_gate_pos_b.norm(dim=-1, keepdim=True)
        self.racing_progress = curr_gate_distance - self.curr_gate_distance
        self.curr_gate_distance = curr_gate_distance

    def _compute_observation(self):
        observation = TensorDict({}, [self.num_envs])
        
        observation["policy"] = torch.cat([
            self.drone.data.root_lin_vel_b,
            self.drone.data.root_ang_vel_b,
            self.drone.data.projected_gravity_b,
            self.drone.multirotor_data.throttle["rotor"],
            self.curr_gate_pos_b,
            self.next_gate_pos_b,
        ], dim=-1)
        return observation
    
    class progress(mdp.RewardFunc):
        env: "RacingSimple"
        def compute(self):
            return self.env.racing_progress
    
    class survival(mdp.RewardFunc):
        def compute(self):
            return torch.ones(self.env.num_envs, 1, device=self.env.device)
    
    class height_exceeds_range(mdp.TerminationFunc):
        env: "RacingSimple"
        def __init__(self, env: "RacingSimple", range):
            super().__init__(env)
            self.range = range

        def compute(self):
            return (
                (self.env.drone.data.root_pos_w[:, 2] < self.range[0]) | 
                (self.env.drone.data.root_pos_w[:, 2] > self.range[1])
            ).unsqueeze(-1)
