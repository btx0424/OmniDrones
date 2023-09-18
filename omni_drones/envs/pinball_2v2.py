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


def encode_drone_pos(drone_pos: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        drone_pos (torch.Tensor): (E,4,3)

    Returns:
        torch.Tensor: (E,4,9) 9=teammate(3)+two opponents(6)
    """
    return torch.stack(
        [
            torch.concat(
                [
                    drone_pos[..., 1, :] - drone_pos[..., 0, :],
                    drone_pos[..., 2, :] - drone_pos[..., 0, :],
                    drone_pos[..., 3, :] - drone_pos[..., 0, :],
                ],
                dim=-1,
            ),
            torch.concat(
                [
                    drone_pos[..., 0, :] - drone_pos[..., 1, :],
                    drone_pos[..., 2, :] - drone_pos[..., 1, :],
                    drone_pos[..., 3, :] - drone_pos[..., 1, :],
                ],
                dim=-1,
            ),
            torch.concat(
                [
                    drone_pos[..., 3, :] - drone_pos[..., 2, :],
                    drone_pos[..., 0, :] - drone_pos[..., 2, :],
                    drone_pos[..., 1, :] - drone_pos[..., 2, :],
                ],
                dim=-1,
            ),
            torch.concat(
                [
                    drone_pos[..., 2, :] - drone_pos[..., 3, :],
                    drone_pos[..., 0, :] - drone_pos[..., 3, :],
                    drone_pos[..., 1, :] - drone_pos[..., 3, :],
                ],
                dim=-1,
            ),
        ],
        dim=1,
    )


def encode_drone_vel(drone_vel: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        drone_vel (torch.Tensor): (E,4,6)

    Returns:
        torch.Tensor: (E,4,18) 9=teammate(6)+two opponents(12)
    """
    return torch.stack(
        [
            torch.concat(
                [
                    drone_vel[..., 1, :],
                    drone_vel[..., 2, :],
                    drone_vel[..., 3, :],
                ],
                dim=-1,
            ),
            torch.concat(
                [
                    drone_vel[..., 0, :],
                    drone_vel[..., 2, :],
                    drone_vel[..., 3, :],
                ],
                dim=-1,
            ),
            torch.concat(
                [
                    drone_vel[..., 3, :],
                    drone_vel[..., 0, :],
                    drone_vel[..., 1, :],
                ],
                dim=-1,
            ),
            torch.concat(
                [
                    drone_vel[..., 2, :],
                    drone_vel[..., 0, :],
                    drone_vel[..., 1, :],
                ],
                dim=-1,
            ),
        ],
        dim=1,
    )


def turn_convert(t: torch.Tensor):
    """convert representation of drone turn

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env,4)  res[:,i]=1.0 if t[:]==i else -1.0
    """
    table = torch.tensor(
        [
            [1.0, 1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ],
        device=t.device,
    )
    return table[t.long()]


def calculate_safety_cost(drone_rpos: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        drone_rpos (torch.Tensor): (E,4,9)

    Returns:
        torch.Tensor: (E,4)
    """
    # r (E,4,3)
    r1, r2, r3 = torch.split(drone_rpos, [3, 3, 3], dim=-1)

    # d (E,4,3)
    d = torch.stack(
        [
            torch.norm(r1, dim=-1),
            torch.norm(r2, dim=-1),
            torch.norm(r3, dim=-1),
        ],
        dim=-1,
    )
    #
    d_m, _ = torch.min(d, dim=-1)

    safety_cost = 1.0 - d_m
    safety_cost = safety_cost.clip(0.0)
    return safety_cost


def out_of_bounds(pos: torch.Tensor, L: float, W: float) -> torch.Tensor:
    return (
        (pos[..., 0] < (-L / 2))
        | (pos[..., 0] > (L / 2))
        | (pos[..., 1] < (-W / 2))
        | (pos[..., 1] > (W / 2))
    )


def in_half_court_0(pos: torch.Tensor, L: float, W: float) -> torch.Tensor:
    return (
        (pos[..., 0] > (-L / 2))
        & (pos[..., 0] < 0)
        & (pos[..., 1] > (-W / 2))
        & (pos[..., 1] < (W / 2))
    )


def in_half_court_1(pos: torch.Tensor, L: float, W: float) -> torch.Tensor:
    return (
        (pos[..., 0] > 0)
        & (pos[..., 0] < (L / 2))
        & (pos[..., 1] > (-W / 2))
        & (pos[..., 1] < (W / 2))
    )


class PingPong2v2(IsaacEnv):
    def __init__(self, cfg: DictConfig, headless: bool):
        self.time_encoding: bool = cfg.task.get("time_encoding", False)
        super().__init__(cfg, headless)

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

        # (n_envs,) False/True(or 0/1) indicating team # needs to toss the ball to the other team
        self.team_turn = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self.last_ball_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)

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
        drone_state_dim = self.drone.state_spec.shape[-1]  # 23
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

        if self.time_encoding:
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

    def _reset_idx(self, env_ids: torch.Tensor):
        """_summary_

        Args:
            env_ids (torch.Tensor): (n_envs_to_reset,)
        """

        # reset drones
        self.drone._reset_idx(env_ids, self.training)
        # drone_pos (len(env_ids),n_drones,3)
        drone_pos = self.init_drone_pos_dist.sample(env_ids.shape)
        drone_rpy = self.init_drone_rpy_dist.sample((*env_ids.shape, 4))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(
            torch.zeros(len(env_ids), 4, 6, device=self.device), env_ids
        )

        # reset the ball

        team_turn = torch.randint_like(env_ids, low=0, high=2) == 0
        # team_turn = torch.zeros_like(env_ids)
        self.team_turn[env_ids] = team_turn

        drone_to_toss_ball_index = team_turn.long() * 2 + torch.randint(
            low=0, high=2, size=(len(env_ids),), device=self.device
        )
        ball_pos_offset = self.init_ball_offset_dist.sample(env_ids.shape)
        ball_pos = (
            drone_pos[torch.arange(len(env_ids)), drone_to_toss_ball_index, :]
            + ball_pos_offset
        )
        ball_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
            len(env_ids), 1
        )
        self.ball.set_world_poses(
            ball_pos + self.envs_positions[env_ids], ball_rot, env_ids
        )
        self.ball.set_velocities(
            torch.zeros(len(env_ids), 6, device=self.device), env_ids
        )
        # fix the mass now
        # ball_masses = self.ball_mass_dist.sample(env_ids.shape)
        ball_masses = torch.ones_like(env_ids) * 0.05
        self.ball.set_masses(ball_masses, env_ids)

        self.stats[env_ids] = 0.0
        self.last_ball_vel[env_ids] = 0.0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        """Comment on observation

        For each drone, the observation vector is of the following format:

        root_state(23)

        relative position of the teammate(3)

        relative position of the two opponents(6)

        velocity of the teammate(6)

        velocity of the two opponents(12)

        relative position of the ball(3)

        turn(1)

        """
        self.root_state = self.drone.get_state()  # (n_envs, 4, 23)
        self.info["drone_state"][:] = self.root_state[..., :13]
        self.ball_pos, ball_rot = self.get_env_poses(self.ball.get_world_poses())
        self.ball_vel = self.ball.get_velocities()[..., :3]
        normalized_ball_vel = self.ball_vel / self.ball_vel.norm(
            dim=-1, keepdim=True
        ).clip(1e-7)
        self.hit = (self.last_ball_vel * normalized_ball_vel).sum(dim=-1) < -0.1
        self.last_ball_vel[:] = normalized_ball_vel

        # self.ball_pos (E,1,3)
        self.rpos_ball = self.ball_pos - self.drone.pos  # (E,4,3)

        self.rpos_drone = encode_drone_pos(self.drone.pos)  # (E,4,9)
        # self.drone.vel (E,4,6)
        encoded_drone_vel = encode_drone_vel(self.drone.vel)  # (E,4,18)

        obs = [
            self.root_state,  # [E, A, *] (E,4,23)
            self.rpos_drone,  # (E,4,9)
            encoded_drone_vel,  # (E,4,18)
            self.rpos_ball,  # [E, A, 3]  (E,4,3)
            self.ball_vel.expand(-1, 4, 3),  # [E, 1, 3] -> [E, A, 3] , (E,4,3)
            turn_convert(self.team_turn).unsqueeze(-1),  # (E,4,1)
        ]

        state = [
            self.root_state.flatten(-2),
            self.rpos_drone.flatten(-2),
            self.rpos_ball.flatten(-2),
            self.ball_vel.flatten(-2),
        ]

        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs.append(t.expand(-1, 4, self.time_encoding_dim))
            state.append(t.squeeze(1))
        obs = torch.cat(obs, dim=-1)
        state = torch.cat(state, dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        hit = self.ball.get_net_contact_forces().any(-1)  # (E,1)

        drone_ball_dist: torch.Tensor = self.rpos_ball.norm(dim=-1)  # (E,4)
        which_drone = drone_ball_dist.argmin(dim=1, keepdim=True)  # (E,1)
        which_team = which_drone / 2  # (E,1)

        # drone of the correct team hit the ball
        switch_turn: torch.Tensor = hit & (
            self.team_turn.unsqueeze(-1).long() == which_team
        )  # (E,1)
        self.team_turn = torch.logical_xor(self.team_turn, switch_turn.squeeze(-1))

        self.stats["team_0_score"].add_(
            (self.team_turn.unsqueeze(-1) & switch_turn).float()
        )
        self.stats["team_1_score"].add_(
            ((~self.team_turn.unsqueeze(-1)) & switch_turn).float()
        )

        ball_too_low = self.ball_pos[..., 2] < 0.2  # (E,1)
        ball_out_of_bounds = out_of_bounds(self.ball_pos, self.L, self.W)  # (E,1)

        # 把球打到了对方的半场，或者对方没有打到自己半场
        # 目前没有考虑打到网上！！！因为触网逻辑比较复杂
        # 目前没有考虑队友之间传球
        team_0_win = ball_too_low & (
            (
                in_half_court_1(self.ball_pos, self.L, self.W)
                & self.team_turn.unsqueeze(-1)
            )
            | (
                torch.logical_not(in_half_court_0(self.ball_pos, self.L, self.W))
                & torch.logical_not(self.team_turn).unsqueeze(-1)
            )
        )

        team_1_win = ball_too_low & (
            (
                in_half_court_0(self.ball_pos, self.L, self.W)
                & self.team_turn.unsqueeze(-1)
            )
            | (
                torch.logical_not(in_half_court_1(self.ball_pos, self.L, self.W))
                & torch.logical_not(self.team_turn).unsqueeze(-1)
            )
        )

        # 击中球了
        reward_hit = switch_turn.float() * 20.0

        reward_game = team_0_win.float() * torch.tensor(
            [[1.0, 1.0, -1.0, -1.0]], device=self.device
        ) + team_1_win.float() * torch.tensor(
            [[-1.0, -1.0, 1.0, 1.0]], device=self.device
        )
        reward_game *= 50.0

        reward = reward_game + reward_hit

        misbehave = ball_too_low

        done = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) | misbehave

        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "done": done,
            },
            self.batch_size,
        )
