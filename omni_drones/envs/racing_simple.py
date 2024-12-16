import torch
import torch.nn.functional as F
import omni_drones
import os
import yaml

from tensordict import TensorDict
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from omni.isaac.lab.sensors import ContactSensorCfg, ContactSensor
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils.math import quat_from_angle_axis, quat_rotate_inverse, quat_rotate, quat_from_euler_xyz
import omni.isaac.lab.sim as sim_utils

from omni_drones.envs import IsaacEnv, mdp
from omni_drones.robots.assets import Multirotor, FIREFLY_CFG, HUMMINGBIRD_CFG


class RacingSimple(IsaacEnv):

    DEFAULT_REWARD_CONFIG = {
        "survival": {"weight": 1.0},
        "progress": {"weight": 5.5},
        # "align": {"weight": 1.0},
        "upright": {"weight": 0.2},
        "spin": {"weight": 0.1},
        "forward": {"weight": 0.8},
        "smoothness": {"weight": 0.1},
        "reach": {"weight": 0.5},
    }

    DEFAULT_TERMINATION_CONFIG = {
        "height_exceeds_range": {"range": (0.3, 5.0)},
        "drift_too_far": {"thres": 8.0},
        "pass_through_illegal": {},
        "crash": {},
    }

    def __init__(self, cfg):
        super().__init__(cfg)
        # the `__init__` method invokes `_design_scene` to create the scene
        # after that, all the entities created are managed by `self.scene`
        print(self.scene)

        # let's get the drone entity
        self.drone: Multirotor = self.scene["drone"]
        self.default_init_state = self.drone.data.default_root_state.clone()
        self.contact: ContactSensor = self.scene["contact_sensor"]

        self.gate: RigidObject = self.scene["gate"]
        self.num_gates = self.gate.num_instances
        root_pose = self.gate.data.root_state_w.clone()
        
        track_cfg_path = os.path.join(omni_drones.__path__[0], "data", "UZH_track.yaml")
        with open(track_cfg_path, "r") as f:
            gate_cfg = yaml.safe_load(f)["gates"]
        gate_pos_w = torch.as_tensor([gate["pos"] for gate in gate_cfg]) * 0.6
        gate_rpy_w = torch.as_tensor([gate["ori"] for gate in gate_cfg]) / 180. * torch.pi
        
        root_pose[:, :3] = gate_pos_w
        root_pose[:, 3:7] = quat_from_euler_xyz(*gate_rpy_w.unbind(-1))
        
        # root_pose[:, 2] += 2.0 + 1.5 * torch.rand(len(root_pose), device=self.device)
        
        # theta = torch.linspace(0, torch.pi * 2, self.num_gates + 1)[:-1]
        # root_pose[:, 0] = 3.6 * torch.cos(theta)
        # root_pose[:, 1] = 3.6 * torch.sin(theta)
        # root_pose[:, 3:7] = quat_from_angle_axis(theta - torch.pi/2, torch.tensor([0., 0., 1.]).expand(len(theta), 3))
        from omni.isaac.lab.markers import VisualizationMarkers, GREEN_ARROW_X_MARKER_CFG
        self.arrows = VisualizationMarkers(GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/direction"))
        self.arrows.set_visibility(True)
        self.arrows.visualize(root_pose[:, :3], root_pose[:, 3:7])
        
        self.gate.write_root_state_to_sim(root_pose)
        self.gate_vec = - quat_rotate(
            self.gate.data.root_quat_w,
            torch.tensor([1., 0., 0.], device=self.device).expand(self.num_gates, 3)
        )
        self.passed = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        
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
            
            drone = HUMMINGBIRD_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot",
            )

            contact_sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                history_length=3,
            )

            gate = RigidObjectCfg(
                prim_path=[f"/World/gate_{i}" for i in range(7)],
                spawn=sim_utils.UsdFileCfg(
                    usd_path="/home/btx0424/lab/gate.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                    ),
                    scale=torch.tensor([1.2, 1.2, 1.1]),
                ),
            )
        return SceneCfg(num_envs=self.cfg.num_envs, env_spacing=2.5)

    def _reset_idx(self, env_ids: torch.Tensor):
        init_state = self.default_init_state[env_ids]
        gate_idx = torch.randint(0, 5, (len(env_ids),), device=self.device)
        curr_gate_pos_w = self.gate.data.root_pos_w[gate_idx]
        rpy = torch.zeros(len(env_ids), 3, device=self.device)
        rpy[:, 2].uniform_(-torch.pi, torch.pi)
        init_state[:, :3] = curr_gate_pos_w + self.gate_vec[gate_idx] * (1. + 2. * torch.rand(len(env_ids), 1, device=self.device))
        init_state[:, 3:7] = quat_from_euler_xyz(*rpy.unbind(-1))
        self.drone.write_root_state_to_sim(init_state, env_ids)

        self.curr_gate_distance[env_ids] = (curr_gate_pos_w - init_state[:, :3]).norm(dim=-1, keepdim=True)

        self.gate_idx[env_ids] = gate_idx
        self.passed[env_ids] = False
    
    def _update_common(self):
        drone_pos_w = self.drone.data.root_pos_w
        drone_quat_w = self.drone.data.root_quat_w
        
        self.curr_gate_pos_w = self.gate.data.root_pos_w[self.gate_idx]
        self.next_gate_pos_w = self.gate.data.root_pos_w[(self.gate_idx + 1) % self.num_gates]
        self.last_gate_pos_w = self.gate.data.root_pos_w[(self.gate_idx - 1) % self.num_gates]
        self.curr_gate_pos_b = quat_rotate_inverse(drone_quat_w, self.curr_gate_pos_w - drone_pos_w)
        self.next_gate_pos_b = quat_rotate_inverse(drone_quat_w, self.next_gate_pos_w - drone_pos_w)
        self.last_gate_pos_b = quat_rotate_inverse(drone_quat_w, self.last_gate_pos_w - drone_pos_w)
        self.curr_gate_vec_w = self.gate_vec[self.gate_idx]
        self.curr_gate_vec_b = quat_rotate_inverse(drone_quat_w, self.curr_gate_vec_w)
        self.last_gate_vec_w = self.gate_vec[(self.gate_idx - 1) % self.num_gates]
        self.last_gate_vec_b = quat_rotate_inverse(drone_quat_w, self.last_gate_vec_w)
        # self.curr_gate_pos_b = self.curr_gate_pos_w - drone_pos_w
        # self.next_gate_pos_b = self.next_gate_pos_w - drone_pos_w
        # self.curr_gate_vec_b = self.gate_vec[self.gate_idx]

        curr_gate_distance = self.curr_gate_pos_b.norm(dim=-1, keepdim=True)
        self.racing_progress = self.curr_gate_distance - curr_gate_distance
        self.curr_gate_distance = curr_gate_distance

        curr_gate_drone = self.curr_gate_pos_w - drone_pos_w
        self.projected_distance = (curr_gate_drone * self.gate_vec[self.gate_idx]).sum(-1)
        self.projected_pos_w = self.curr_gate_pos_w - self.projected_distance.unsqueeze(-1) * self.curr_gate_vec_w

        passed = self.projected_distance > 0
        paass_through = (~self.passed) & passed
        self.pass_through_legal_ = paass_through & (curr_gate_distance < 0.4).squeeze(-1)
        self.pass_through_illegal_ = paass_through & (curr_gate_distance >= 0.4).squeeze(-1)
        self.passed = passed

        self.gate_idx = torch.where(self.pass_through_legal_, (self.gate_idx + 1) % self.num_gates, self.gate_idx)

    def _compute_observation(self):
        observation = TensorDict({}, [self.num_envs])
        target = torch.tensor([0., 0., 1.5], device=self.device)

        observation[("agents", "observation")] = torch.cat([
            self.drone.data.root_lin_vel_b,
            self.drone.data.root_ang_vel_b,
            self.drone.data.root_ang_vel_w[:, :2],
            self.drone.data.root_quat_w,
            self.drone.data.root_pos_w[:, 2].unsqueeze(1),
            self.drone.data.projected_gravity_b,
            self.drone.multirotor_data.throttle["rotor"],
            self.drone.multirotor_data.throttle_change["rotor"],
            self.last_gate_pos_b,
            self.curr_gate_pos_b,
            self.curr_gate_pos_b.norm(dim=-1, keepdim=True),
            self.projected_distance.unsqueeze(-1),
            self.curr_gate_vec_b,
            self.next_gate_pos_b,
        ], dim=-1)
        return observation
    
    class progress(mdp.RewardFunc):
        env: "RacingSimple"
        def compute(self):
            return self.env.racing_progress + self.env.pass_through_legal_.unsqueeze(1) * 30.
    
    class reach(mdp.RewardFunc):
        env: "RacingSimple"
        def compute(self):
            # target = torch.tensor([0., 0., 1.5], device=self.env.device)
            target = self.env.curr_gate_pos_w - self.env.curr_gate_vec_w * 0.2
            distance = (self.env.drone.data.root_pos_w[:, :3] - target).square().sum(-1, True)
            return torch.exp(- distance / 0.5)
    
    class forward(mdp.RewardFunc):
        env: "RacingSimple"
        def compute(self):
            direction = F.normalize(self.env.curr_gate_pos_b[:, :2], dim=-1)
            return -direction[:, 0].unsqueeze(-1)
    
    class align(mdp.RewardFunc):
        env: "RacingSimple"
        def compute(self):
            d1 = (self.env.projected_pos_w - self.env.drone.data.root_pos_w).norm(dim=-1, keepdim=True)
            d2 = self.env.projected_distance.unsqueeze(-1)
            return - (d1 - d2).clamp(0.)

    class upright(mdp.RewardFunc):
        env: "RacingSimple"
        def compute(self):
            return - self.env.drone.data.projected_gravity_b[:, :2].square().sum(-1, True)
    
    class spin(mdp.RewardFunc):
        env: "RacingSimple"
        def compute(self):
            return - self.env.drone.data.root_ang_vel_w.square().sum(-1, True)
        
    class smoothness(mdp.RewardFunc):
        env: "RacingSimple"
        def compute(self):
            return - self.env.drone.multirotor_data.throttle_change["rotor"].square().sum(-1, True)

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
    
    class pass_through_illegal(mdp.TerminationFunc):
        env: "RacingSimple"
        def compute(self):
            return self.env.pass_through_illegal_.unsqueeze(-1)

    class drift_too_far(mdp.TerminationFunc):
        env: "RacingSimple"
        def __init__(self, env: "RacingSimple", thres):
            super().__init__(env)
            self.thres = thres

        def compute(self):
            return (self.env.curr_gate_distance > self.thres).reshape(self.num_envs, -1)

    class crash(mdp.TerminationFunc):
        env: "RacingSimple"
        def compute(self):
            return self.env.contact.data.net_forces_w.norm(dim=-1).any(dim=1, keepdim=True)
            

    def debug_vis(self):
        self.debug_draw.vector(
            self.drone.data.root_pos_w[0],
            self.curr_gate_pos_w[0] - self.curr_gate_vec_w[0] * 0.2 - self.drone.data.root_pos_w[0],
            color=(0, 1, 0, 1)
        )
        self.debug_draw.vector(
            self.curr_gate_pos_w[0] - self.curr_gate_vec_w[0] * 0.2,
            self.next_gate_pos_w[0] - (self.curr_gate_pos_w[0] - self.curr_gate_vec_w[0] * 0.2),
            color=(0, 1, 0, 1)
        )
        self.debug_draw.vector(
            self.drone.data.root_pos_w[0],
            self.projected_pos_w[0] - self.drone.data.root_pos_w[0],
            color=(1, 0, 0, 1)
        )

