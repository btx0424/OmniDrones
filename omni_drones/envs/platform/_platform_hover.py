from functorch import vmap
import torch
import torch.distributions as D

import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.prims as prim_utils
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

import omni_drones.utils.kit as kit_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import cpos, off_diag, others
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.scene import design_scene
from omni_drones.utils.torch import euler_to_quaternion

from .utils import OveractuatedPlatform, PlatformCfg


def compose_transform(
    parent_translation: torch.Tensor,
    parent_rotation: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
):
    rotation = torch_utils.quat_mul(
        parent_rotation.expand_as(rotation), 
        rotation
    )
    translation = parent_translation + torch_utils.quat_rotate(rotation, translation)
    return translation, rotation


class PlatformHover(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = self.cfg.task.reward_action_smoothness_weight
        self.reward_potential_weight = self.cfg.task.reward_potential_weight
        self.reward_distance_scale = self.cfg.task.reward_distance_scale

        self.platform.initialize()

        self.target_vis = RigidPrimView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        ).initialize()
        
        self.init_vels = torch.zeros_like(self.platform.get_velocities())
        self.init_joint_pos = self.platform.get_joint_positions(clone=True)
        self.init_joint_vel = torch.zeros_like(self.platform.get_joint_velocities())

        drone_state_dim = self.drone.state_spec.shape.numel()
        observation_spec = CompositeSpec({
            "state_self": UnboundedContinuousTensorSpec((1, drone_state_dim + self.drone.n)),
            "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, 13)),
            "state_frame": UnboundedContinuousTensorSpec((1, 25)),
        }).to(self.device)
        state_spec = CompositeSpec({
            "state_drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim + self.drone.n)),
            "state_frame": UnboundedContinuousTensorSpec((1, 25)),
        }).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.drone.n,
            observation_spec,
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
            state_spec
        )

        self.init_pos_dist = D.Uniform(
            torch.tensor([-3., -3., 1.5], device=self.device),
            torch.tensor([3.0, 3.0, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2], device=self.device) * torch.pi
        )
        self.target_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.0], device=self.device),
            torch.tensor([0.2, 0.2, 2.0], device=self.device)
        )
        self.target_pos = torch.tensor([0., 0., 2.25], device=self.device)
        self.target_heading =  torch.zeros(self.num_envs, 3, device=self.device)
        self.target_up = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_distance = torch.zeros(self.num_envs, 1, device=self.device)

        self.alpha = 0.7
        stats_spec = CompositeSpec({
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "effort": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec((1))
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        platform_cfg = PlatformCfg(
            num_drones=self.cfg.task.num_drones,
            arm_length=self.cfg.task.arm_length,
            joint_damping=self.cfg.task.joint_damping
        )
        self.platform = OveractuatedPlatform(
            cfg=platform_cfg,
            drone=self.drone,
        )
        self.platform.spawn(
            translations=[0., 0., 2.],
            enable_collision=True
        )

        # for visulization
        target_prim_path = self.platform._create_frame(
            "/World/envs/env_0/target",
            enable_collision=False
        ).GetPath().pathString
        kit_utils.set_rigid_body_properties(target_prim_path, disable_gravity=True)

        design_scene()
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        platform_pos = self.init_pos_dist.sample(env_ids.shape) 
        platform_rpy = self.init_rpy_dist.sample(env_ids.shape)
        platform_rot = euler_to_quaternion(platform_rpy)
        platform_heading = torch_utils.quat_axis(platform_rot, 0)
        platform_up = torch_utils.quat_axis(platform_rot,  2)
        self.platform.set_world_poses(
            platform_pos + self.envs_positions[env_ids], 
            platform_rot, env_indices=env_ids
        )
        self.platform.set_velocities(self.init_vels[env_ids], env_ids)

        self.platform.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.platform.set_joint_velocities(self.init_joint_vel[env_ids], env_ids)

        target_rpy = self.target_rpy_dist.sample(env_ids.shape)
        target_rot = euler_to_quaternion(target_rpy)
        target_heading = torch_utils.quat_axis(target_rot, 0)
        target_up = torch_utils.quat_axis(target_rot, 2)
        self.target_heading[env_ids] = target_heading
        self.target_up[env_ids] = target_up
        
        self.target_vis.set_world_poses(
            self.target_pos + self.envs_positions[env_ids],
            orientations=target_rot,
            env_indices=env_ids
        )

        self.stats[env_ids] = 0.
        distance = torch.cat([
            self.target_pos - platform_pos,
            target_heading - platform_heading,
            target_up - platform_up
        ], dim=-1).norm(dim=-1, keepdim=True)
        self.last_distance[env_ids] = distance

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        drone_pos = self.drone_states[..., :3]
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)

        self.platform_pos, platform_rot = self.get_env_poses(self.platform.get_world_poses(clone=True))
        self.platform_heading = vmap(torch_utils.quat_axis)(platform_rot, axis=0)
        self.platform_up = vmap(torch_utils.quat_axis)(platform_rot, axis=2)
        self.platform_vels = self.platform.get_velocities(clone=True)

        self.target_platform_rpos = self.target_pos - self.platform_pos
        self.target_platform_rheading = self.target_heading.unsqueeze(1) - self.platform_heading
        self.target_platform_rup = self.target_up.unsqueeze(1) - self.platform_up
        self.target_platform_rpose = torch.cat([
            self.target_platform_rpos,
            self.target_platform_rheading,
            self.target_platform_rup
        ], dim=-1)

        platform_drone_rpos = self.platform_pos - self.drone_states[..., :3]
        platform_state = torch.cat([
            self.target_platform_rpose, # 9
            platform_rot, # 4
            self.platform_heading, # 3
            self.platform_up, # 3
            self.platform_vels, # 6
        ], dim=-1) # [num_envs, 1, 25]


        identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)

        obs = TensorDict({}, [self.num_envs, self.drone.n])
        obs["state_self"] = torch.cat(
            [-platform_drone_rpos, self.drone_states[..., 3:], identity], dim=-1
        ).unsqueeze(2)
        obs["state_others"] = torch.cat(
            [self.drone_rpos, vmap(others)(self.drone_states[..., 3:13])], dim=-1
        )
        obs["state_frame"] = platform_state.unsqueeze(1).expand(-1, self.drone.n, 1, -1)

        state = TensorDict({}, [self.num_envs])
        state["state_drones"] = obs["state_self"].squeeze(2)    # [num_envs, drone.n, drone_state_dim]
        state["state_frame"] = platform_state                # [num_envs, 1, platform_state_dim]
        
        pos_error = torch.norm(self.target_platform_rpos, dim=-1)
        heading_alignment = torch.sum(self.platform_heading * self.target_heading.unsqueeze(1), dim=-1)
        
        self.stats["pos_error"].lerp_(pos_error, (1-self.alpha))
        self.stats["heading_alignment"].lerp_(heading_alignment, (1-self.alpha))
        self.stats["action_smoothness"].add_(-self.drone.throttle_difference.mean(-1, True))

        return TensorDict(
            {
                "drone.obs": obs,
                "drone.state": state,
                "stats": self.stats
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        platform_vels = self.platform.get_velocities()

        distance = torch.norm(self.target_platform_rpose, dim=-1)
        
        reward = torch.zeros(self.num_envs, self.drone.n, 1, device=self.device)
        # reward_pose = 1 / (1 + torch.square(distance * self.reward_distance_scale))
        reward_pose = torch.exp(- self.reward_distance_scale * distance)
        
        up = torch.sum(self.platform_up * self.target_up.unsqueeze(1), dim=-1)
        reward_up = torch.square((up + 1) / 2)

        spinnage = platform_vels[:, -3:].abs().sum(-1)
        reward_spin = 1. / (1 + torch.square(spinnage))
        
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort).mean(-1, keepdim=True)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference).mean(-1, keepdim=True)

        reward_potential = self.last_distance - distance
        self.last_distance[:] = distance
        assert reward_pose.shape == reward_up.shape == reward_action_smoothness.shape

        reward[:] = (
            reward_pose
            + reward_pose * (reward_up + reward_spin) 
            + reward_potential
            + reward_effort
            + reward_action_smoothness
        ).unsqueeze(1)

        done_misbehave = (self.drone_states[..., 2] < 0.2).any(-1, keepdim=True) | (distance > 5.0)
        done_hasnan = done_hasnan = torch.isnan(self.drone_states).any(-1)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | done_misbehave
            | done_hasnan.any(-1, keepdim=True)
        )

        self._tensordict["return"] += reward
        return TensorDict(
            {
                "reward": {"drone.reward": reward},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )

