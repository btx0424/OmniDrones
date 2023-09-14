from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omegaconf import DictConfig

from omni_drones.views import RigidPrimView
from omni_drones.robots.drone import MultirotorBase
import omni_drones.utils.kit as kit_utils

from omni_drones.utils.torch import euler_to_quaternion
from tensordict.tensordict import TensorDict, TensorDictBase
import omni.isaac.core.objects as objects
import omni.isaac.core.materials as materials

import torch
from torch import Tensor
import torch.distributions as D
import torch.functional as F
import torch.nn.functional as NNF

from pxr import UsdShade
from typing import List, Dict

from torchrl.data import (
    UnboundedContinuousTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
)


def get_dist(dist_cfg: Dict, device):
    low = torch.tensor(dist_cfg["low"], device=device)
    high = torch.tensor(dist_cfg["high"], device=device)
    return D.Uniform(
        low=low,
        high=high,
    )


def get_extended_pos_dist(
    x_low: float,
    y_low: float,
    z_low: float,
    x_high: float,
    y_high: float,
    z_high: float,
    device,
):
    return D.Uniform(
        torch.tensor(
            [
                [-x_high, -y_high, z_low],
                [-x_high, y_low, z_low],
                [x_low, y_low, z_low],
                [x_low, -y_high, z_low],
            ],
            device=device,
        ),
        torch.tensor(
            [
                [-x_low, -y_low, z_high],
                [-x_low, y_high, z_high],
                [x_high, y_high, z_high],
                [x_high, -y_low, z_high],
            ],
            device=device,
        ),
    )


class PingPong2v2(IsaacEnv):
    def __init__(self, cfg: DictConfig, headless: bool):
        super().__init__(cfg, headless)

        self.time_encoding: bool = self.cfg.task.get("time_encoding", False)

        self.drone.initialize()
        self.ball = RigidPrimView(
            "/World/envs/env_*/ball",
            reset_xform_properties=False,
            track_contact_forces=True,
            shape=(-1, 1),
        )
        self.ball.initialize()

        task_cfg: DictConfig = cfg.task

        self.L: float = task_cfg.court.L * task_cfg.court.multiplier
        self.W: float = task_cfg.court.W * task_cfg.court.multiplier

        self.init_ball_offset_dist = get_dist(
            task_cfg.initial.ball_offset_dist, device=self.device
        )

        self.init_drone_pos_dist = get_extended_pos_dist(
            *task_cfg.initial.drone_xyz_dist.low,
            *task_cfg.initial.drone_xyz_dist.high,
            device=self.device
        )
        self.init_drone_rpy_dist = get_dist(
            task_cfg.initial.drone_rpy_dist, device=self.device
        )  # unit: \pi

        # (n_envs,) indicating whether it's team 0's turn to toss the ball
        self.team_turn = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        material = materials.PhysicsMaterial(
            prim_path="/World/Physics_Materials/physics_material_0",
            restitution=0.8,
        )
        objects.DynamicSphere(
            prim_path="/World/envs/env_0/ball",
            radius=0.05,
            mass=0.05,
            color=torch.tensor([1.0, 0.2, 0.2]),
            physics_material=material,
        )
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        # placeholders
        drone_prims = self.drone.spawn(
            translations=[
                (1.0, -1.0, 1.0),
                (1.0, 1.0, 2.0),
                (-1.0, -1.0, 3.0),
                (-1.0, 1.0, 4.0),
            ]
        )

        material = UsdShade.Material(material.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("base_link/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1] # 23
        observation_dim = (
            drone_state_dim
            + 3
            + 3  # ball's position and linear velocity
            + 9
            + 18  # other's position and linear and angular velocity
            + 1  # turn
        )  # 57
        state_dim = (
            drone_state_dim * 4
            + 12
            + 3  # relative ball pos, ball vel
            + 12  # relative drone pos,
            + 4  # turn
        )  # 123

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim
            state_dim += self.time_encoding_dim

        self.observation_spec = (
            CompositeSpec(
                {
                    "agents": {
                        "observation": UnboundedContinuousTensorSpec(
                            (4, observation_dim)
                        ),
                        "state": UnboundedContinuousTensorSpec(state_dim),
                    }
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )

        self.action_spec = (
            CompositeSpec(
                {
                    "agents": {
                        "action": torch.stack(
                            [self.drone.action_spec] * self.drone.n, dim=0
                        ),
                    }
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.reward_spec = (
            CompositeSpec({"agents": {"reward": UnboundedContinuousTensorSpec((4, 1))}})
            .expand(self.num_envs)
            .to(self.device)
        )

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            4,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state"),
        )

        stats_spec = (
            CompositeSpec(
                {
                    "episode_len": UnboundedContinuousTensorSpec(1),
                    "team_0_score": UnboundedContinuousTensorSpec(1),
                    "team_1_score": UnboundedContinuousTensorSpec(1),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        info_spec = (
            CompositeSpec(
                {
                    "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()
