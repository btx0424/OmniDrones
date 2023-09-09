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
import torch.distributions as D
import torch.functional as F
import torch.nn.functional as NNF

from pxr import UsdShade
from typing import List

from torchrl.data import (
    UnboundedContinuousTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
)

from omni.isaac.debug_draw import _debug_draw


def turn_convert(t: torch.Tensor):
    """convert representation of drone turn

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env,4)  res[:,i]=1.0 if t[:]==i else -1.0
    """
    table = torch.tensor(
        [
            [1.0, -1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0, 1.0],
        ],
        device=t.device,
    )
    return table[t]


def turn_shift(t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        t (torch.Tensor): (n_env,) int64
        h (torch.Tensor): (n_env,) bool

    Returns:
        torch.Tensor: (n_env,) int64
    """
    return (t + h.long()) % 4


# @torch.jit.script
def encode_drone_pos(drone_pos: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        drone_pos (torch.Tensor): (E,4,3)

    Returns:
        torch.Tensor: (E,4,9)
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
                    drone_pos[..., 2, :] - drone_pos[..., 1, :],
                    drone_pos[..., 3, :] - drone_pos[..., 1, :],
                    drone_pos[..., 0, :] - drone_pos[..., 1, :],
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
                    drone_pos[..., 0, :] - drone_pos[..., 3, :],
                    drone_pos[..., 1, :] - drone_pos[..., 3, :],
                    drone_pos[..., 2, :] - drone_pos[..., 3, :],
                ],
                dim=-1,
            ),
        ],
        dim=1,
    )


# @torch.jit.script
def encode_drone_vel(drone_vel: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        drone_vel (torch.Tensor): (E,4,6)

    Returns:
        torch.Tensor: (E,4,18)
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
                    drone_vel[..., 2, :],
                    drone_vel[..., 3, :],
                    drone_vel[..., 0, :],
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
                    drone_vel[..., 0, :],
                    drone_vel[..., 1, :],
                    drone_vel[..., 2, :],
                ],
                dim=-1,
            ),
        ],
        dim=1,
    )


# @torch.jit.script
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


class PingPongRelay(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)

        self.time_encoding = self.cfg.task.time_encoding

        self.drone.initialize()
        randomization = self.cfg.task.get("randomization", None)
        if randomization is not None:
            if "drone" in self.cfg.task.randomization:
                self.drone.setup_randomization(self.cfg.task.randomization["drone"])

        self.ball = RigidPrimView(
            "/World/envs/env_*/ball",
            reset_xform_properties=False,
            track_contact_forces=True,
            shape=(-1, 1),
        )

        self.ball.initialize()

        self.ball_mass_dist = D.Uniform(
            torch.tensor([0.01], device=self.device),
            torch.tensor([0.05], device=self.device),
        )
        self.init_ball_offset_dist = D.Uniform(
            torch.tensor([-0.3, -0.3, 1.6], device=self.device),
            torch.tensor([0.3, 0.3, 2.0], device=self.device),
        )

        z_low, z_high = 1.5, 1.7
        x_low, x_high = 0.5, 1.7
        y_low, y_high = 0.5, 1.7

        self.init_drone_pos_dist = D.Uniform(
            torch.tensor(
                [
                    [-x_high, -y_high, z_low],
                    [-x_high, y_low, z_low],
                    [x_low, y_low, z_low],
                    [x_low, -y_high, z_low],
                ],
                device=self.device,
            ),
            torch.tensor(
                [
                    [-x_low, -y_low, z_high],
                    [-x_low, y_high, z_high],
                    [x_high, y_high, z_high],
                    [x_high, -y_low, z_high],
                ],
                device=self.device,
            ),
        )
        self.init_drone_rpy_dist = D.Uniform(
            torch.tensor([-0.1, -0.1, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, 2.0], device=self.device) * torch.pi,
        )

        # turn[env_id] \in {0,1,2,3}. drone turn[env_id] should relay the ball to (turn[env_id]+1)%4
        self.turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)

        self.last_ball_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.draw = _debug_draw.acquire_debug_draw_interface()

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
        drone_state_dim = self.drone.state_spec.shape[-1]
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
                    "return": UnboundedContinuousTensorSpec(1),
                    "episode_len": UnboundedContinuousTensorSpec(1),
                    "score": UnboundedContinuousTensorSpec(1),
                    # 0: cummulative deviation 1: count(score)
                    "angular_deviation": UnboundedContinuousTensorSpec(2),
                    "active_velocity": UnboundedContinuousTensorSpec(1),
                    "inactive_velocity": UnboundedContinuousTensorSpec(3),
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
        self.drone._reset_idx(env_ids, self.training)

        # drone_pos (n_env,n_drones,3)
        drone_pos = self.init_drone_pos_dist.sample(env_ids.shape)
        drone_rpy = self.init_drone_rpy_dist.sample((*env_ids.shape, 4))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(
            torch.zeros(len(env_ids), 4, 6, device=self.device), env_ids
        )

        turn = torch.randint(0, 4, (len(env_ids),), device=self.device)
        self.turn[env_ids] = turn

        ball_pos_offset = self.init_ball_offset_dist.sample(env_ids.shape)
        ball_pos = drone_pos[torch.arange(len(env_ids)), turn, :] + ball_pos_offset

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
        self.last_ball_vel[env_ids] = 0.0
        self.stats[env_ids] = 0.0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()  # (,23)
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
            turn_convert(self.turn).unsqueeze(-1),  # (E,4,1)
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

        if False:
            central_env_pos = self.envs_positions[self.central_env_idx]
            drone_pos = self.drone.pos[self.central_env_idx]
            ball_pos = self.ball_pos[self.central_env_idx]
            turn = self.turn[self.central_env_idx]
            point_list_0 = (drone_pos[turn] + central_env_pos).tolist()
            point_list_1 = (ball_pos + central_env_pos).tolist()
            colors = [(1.0, 1.0, 0.5, 0.7)]
            sizes = [1.2]
            self.draw.clear_lines()
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

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
        which_drone = self.rpos_ball.norm(dim=-1).argmin(dim=1, keepdim=True)  # (E,1)
        switch_turn: torch.Tensor = hit & (
            self.turn.unsqueeze(-1) == which_drone
        )  # (E,1)
        vxy_ball = self.ball_vel[:, 0, :2]  # (E,2)
        rposxy = self.rpos_drone[torch.arange(self.num_envs), self.turn, :2]  # (E,2)
        cosine_similarity = NNF.cosine_similarity(vxy_ball, rposxy, dim=-1)  # (E,)
        angular_deviation = torch.acos(cosine_similarity)  # (E,)

        self.turn = turn_shift(self.turn, switch_turn.squeeze(-1))
        # (E,4)
        inactive_mask = self.turn.unsqueeze(-1) != torch.arange(
            4, device=self.turn.device
        )

        # 不能离得太近
        safety_cost = calculate_safety_cost(self.rpos_drone)  # (E,4)

        # 接球的无人机要离球近
        reward_pos = (turn_convert(self.turn) == 1).float() / (
            1 + torch.norm(self.rpos_ball[..., :2], dim=-1)
        )  # (E,4)

        # 无人机要在球下面
        reward_height = 0.8 * (
            self.ball_pos[..., 2]
            - self.drone.pos[..., 2].max(dim=1, keepdim=True)[0].clip(1.5)
        ).clip(0.3)

        # 击中球了获得高额奖励
        reward_score = switch_turn.float() * 50.0

        # clamp at about 16 degree
        angular_penalty = (
            switch_turn.squeeze(-1).float()
            * (1 - cosine_similarity).clamp(min=0.04, max=1.0)
            * 30.0
        )
        angular_penalty = angular_penalty.unsqueeze(-1)  # (E,1)

        #
        moving_penalty = inactive_mask.float() * self.root_state[:, :, 6:9].norm(dim=-1)
        moving_penalty *= 0.3

        reward = torch.sum(
            reward_pos
            + reward_height
            + reward_score
            - safety_cost
            - angular_penalty
            - moving_penalty,
            dim=1,
            keepdim=True,
        ).expand(-1, 4)

        misbehave = (
            (self.drone.pos[..., 2] < 0.3).any(-1, keepdim=True)
            | (self.ball_pos[..., 2] < 0.2)
            | (self.ball_pos[..., 2] > 5.0)
            | (self.ball_pos[..., 0].abs() > 2)
            | (self.ball_pos[..., 1].abs() > 2)
        )  # [E, 1]

        done = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) | misbehave

        self.stats["return"].add_(reward[..., [0]])
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["score"].add_(switch_turn.float())

        self.stats["angular_deviation"][switch_turn.squeeze(-1), 0] += (
            angular_deviation[switch_turn.squeeze(-1)] / torch.pi * 180
        )
        self.stats["angular_deviation"][switch_turn.squeeze(-1), 1] += 1

        self.stats["active_velocity"] = self.root_state[
            torch.arange(self.num_envs), self.turn, 6:9
        ].norm(dim=-1, keepdim=True)

        # sele.root_state (E,4,23)
        tmp = self.root_state[inactive_mask].view(self.num_envs, 3, -1)  # (E,3,23)
        self.stats["inactive_velocity"] = tmp[:, :, 6:9].norm(dim=-1)

        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "done": done,
            },
            self.batch_size,
        )
