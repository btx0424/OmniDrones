import torch
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    UnboundedContinuousTensorSpec, 
    CompositeSpec, 
    BinaryDiscreteTensorSpec
)

import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import omni.isaac.core.objects as objects
from omni.isaac.debug_draw import _debug_draw

import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, normalize
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView

from .utils import create_pendulum
from ..utils import create_obstacle

class InvPendulumFlyThrough(IsaacEnv):
    r"""
    An intermidiate control task where a classic inverted pendulum is based on the UAV.
    We refer the the ball at the end of pendulum as *payload*. The goal for the agent 
    is to pass between two horizontal bars and keep balance while maintaining its position 
    around a target position. If the distance between the two horizontal bars is less than 
    the length of the rod, the agent need learn to swing the inverted pendulum to pass through.

    Observation
    -----------
    - `drone_payload_rpos` (3): The position of the drone relative to the payload's position.
    - `root_state` (16 + num_rotors): The basic information of the drone (except its position), 
      containing its rotation (in quaternion), velocities (linear and angular), 
      heading and up vectors, and the current throttle.
    - `target_payload_rpos` (3): The position of the reference relative to the payload's position.
    - `payload_vel` (6): The linear and angular velocities of the payload.
    - `obstacle_drone_rpos` (2 * 2 = 4): The position of the two bars relative to the drone's position.
    - *time_encoding*: 

    Reward 
    ------
    - pos: 
    - bar_up:
    - effort:
    - spin:
    - swing:
    - collision:

    The total reward is 

    .. math:: 
    
        r = [r_\text{pos} + r_\text{pos} * (r_\text{bar_up} + r_\text{spin} + r_\text{swing}) + r_\text{effort}] * (1 - r_\text{collision})

    Episode End
    -----------
    - Termination: 

    Config
    ------

    """
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = self.cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.reset_on_collision = self.cfg.task.reset_on_collision
        self.time_encoding = self.cfg.task.time_encoding
        self.obstacle_spacing = self.cfg.task.obstacle_spacing

        self.drone.initialize()

        self.obstacles = RigidPrimView(
            "/World/envs/env_*/obstacle_*",
            reset_xform_properties=False,
            shape=[self.num_envs, -1],
            track_contact_forces=True
        )
        self.obstacles.initialize()
        self.payload = RigidPrimView(
            f"/World/envs/env_*/{self.drone.name}_*/payload",
        )
        self.payload.initialize()
        self.bar = RigidPrimView(
            f"/World/envs/env_*/{self.drone.name}_*/bar",
        )
        self.bar.initialize()

        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())
        self.obstacle_pos = self.get_env_poses(self.obstacles.get_world_poses())[0]

        drone_state_dim = self.drone.state_spec.shape[-1]
        if self.time_encoding:
            self.time_encoding_dim = 4
            drone_state_dim += self.time_encoding_dim
        
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            UnboundedContinuousTensorSpec(drone_state_dim + 12 + 4).to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )

        self.init_pos_dist = D.Uniform(
            torch.tensor([-2.2, 0, 1.0], device=self.device),
            torch.tensor([-1.0, 0, 2.0], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.15, -.15, 0.], device=self.device) * torch.pi,
            torch.tensor([0.15, 0.15, 2], device=self.device) * torch.pi
        )
        self.obstacle_spacing_dist = D.Uniform(
            torch.tensor(self.obstacle_spacing[0], device=self.device),
            torch.tensor(self.obstacle_spacing[1], device=self.device)
        )
        payload_mass_scale = self.cfg.task.payload_mass_scale
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(payload_mass_scale[0] * self.drone.MASS_0, device=self.device),
            torch.as_tensor(payload_mass_scale[1] * self.drone.MASS_0, device=self.device)
        )
        self.bar_mass_dist = D.Uniform(
            torch.as_tensor(self.cfg.task.bar_mass_min, device=self.device),
            torch.as_tensor(self.cfg.task.bar_mass_max, device=self.device)
        )

        self.alpha = 0.8
        stats_spec = CompositeSpec({
            "pos_error": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "success": BinaryDiscreteTensorSpec(1, dtype=bool),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.payload_traj_vis = []
        self.drone_traj_vis = []

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        create_obstacle(
            "/World/envs/env_0/obstacle_0", 
            prim_type="Capsule",
            translation=(0., 0., 1.2),
            attributes={"axis": "Y", "radius": 0.04, "height": 5}
        )
        create_obstacle(
            "/World/envs/env_0/obstacle_1", 
            prim_type="Capsule",
            translation=(0., 0., 2.2),
            attributes={"axis": "Y", "radius": 0.04, "height": 5}
        )

        self.drone.spawn(translations=[(0.0, 0.0, 2.)])
        create_pendulum(
            f"/World/envs/env_0/{self.drone.name}_0", 
            self.cfg.task.bar_length,
            payload_radius=0.04
        )

        self.payload_target_pos = torch.tensor([1.5, 0., 2.3], device=self.device)
        sphere = objects.DynamicSphere(
            "/World/envs/env_0/target",
            translation=self.payload_target_pos,
            radius=0.05,
            color=torch.tensor([1., 0., 0.])
        )
        kit_utils.set_collision_properties(sphere.prim_path, collision_enabled=False)
        kit_utils.set_rigid_body_properties(sphere.prim_path, disable_gravity=True)
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        
        drone_pos = self.init_pos_dist.sample((*env_ids.shape, 1))
        drone_rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.drone.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.drone.set_joint_velocities(self.init_joint_vels[env_ids], env_ids)

        payload_mass = self.payload_mass_dist.sample(env_ids.shape)
        self.payload.set_masses(payload_mass, env_ids)
        bar_mass = self.bar_mass_dist.sample(env_ids.shape)
        self.bar.set_masses(bar_mass, env_ids)

        obstacle_spacing = self.obstacle_spacing_dist.sample(env_ids.shape)
        obstacle_pos = torch.zeros(len(env_ids), 2, 3, device=self.device)
        obstacle_pos[:, :, 2] = 1.2
        obstacle_pos[:, 1, 2] += obstacle_spacing
        self.obstacles.set_world_poses(
            obstacle_pos + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids
        )
        self.obstacle_pos[env_ids] = obstacle_pos

        self.stats.exclude("success")[env_ids] = 0.
        self.stats["success"][env_ids] = False

        if (env_ids == self.central_env_idx).any():
            self.payload_traj_vis.clear()
            self.drone_traj_vis.clear()
            self.draw.clear_lines()

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.drone_up = self.drone_state[..., 16:19]
        self.payload_pos, payload_rot = self.get_env_poses(self.payload.get_world_poses())
        self.payload_vels = self.payload.get_velocities()

        # relative position and heading
        self.drone_payload_rpos = self.drone_state[..., :3] - self.payload_pos.unsqueeze(1)
        self.target_payload_rpos = self.payload_target_pos - self.payload_pos.unsqueeze(1)
        obstacle_drone_rpos = self.obstacle_pos[..., [0, 2]] - self.drone_state[..., [0, 2]]

        obs = [
            self.drone_payload_rpos, # 3
            self.drone_state,
            self.target_payload_rpos, # 3
            self.payload_vels.unsqueeze(1), # 6
            obstacle_drone_rpos.flatten(start_dim=-2).unsqueeze(1), 
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)
        
        self.pos_error = torch.norm(self.target_payload_rpos, dim=-1)
        self.stats["pos_error"].lerp_(self.pos_error, (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))

        if self._should_render(0):
            central_env_pos = self.envs_positions[self.central_env_idx]
            drone_pos = (self.drone.pos[self.central_env_idx, 0]+central_env_pos).tolist()
            payload_pos = (self.payload_pos[self.central_env_idx]+central_env_pos).tolist()
            
            if len(self.payload_traj_vis)>1:
                point_list_0 = [self.payload_traj_vis[-1], self.drone_traj_vis[-1]]
                point_list_1 = [payload_pos, drone_pos]
                colors = [(1., .1, .1, 1.), (.1, 1., .1, 1.)]
                sizes = [1.5, 1.5]
                self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)
            
            self.drone_traj_vis.append(drone_pos)
            self.payload_traj_vis.append(payload_pos)
        
        return TensorDict({
            "drone.obs": obs,
            "stats": self.stats
        }, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot, vels = self.drone_state[..., :13].split([3, 4, 6], dim=-1)
        
        # pos_reward = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
        pos_reward = torch.exp(-self.reward_distance_scale * self.pos_error)

        bar_reward_up = normalize(-self.drone_payload_rpos)[..., 2]

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)

        spin = torch.square(vels[..., -1])
        reward_spin = 1. / (1.0 + torch.square(spin))

        swing = torch.norm(self.payload_vels[..., :3], dim=-1, keepdim=True)
        reward_swing = 1. * torch.exp(-swing)

        collision = (
            self.obstacles
            .get_net_contact_forces()
            .any(-1)
            .any(-1, keepdim=True)
        )
        collision_reward = collision.float()

        self.stats["collision"].add_(collision_reward)
        assert bar_reward_up.shape == reward_spin.shape == reward_swing.shape
        reward = (
            pos_reward
            + pos_reward * (bar_reward_up + reward_spin + reward_swing) 
            + reward_effort
        ) * (1 - collision_reward)
        
        done_misbehave = (
            (pos[..., 2] < 0.2) 
            | (pos[..., 1].abs() > 2.)
            | (bar_reward_up < 0.2) 
            | (self.payload_pos[:, 2] > 3.).unsqueeze(-1)
        )
        done_hasnan = torch.isnan(self.drone_state).any(-1)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | done_misbehave
            | done_hasnan
        ) 

        if self.reset_on_collision:
            done |= collision

        self.stats["success"].bitwise_or_(self.pos_error < 0.2)
        self._tensordict["return"] += reward.unsqueeze(-1)
        return TensorDict(
            {
                "reward": {"drone.reward": reward.unsqueeze(-1)},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )
