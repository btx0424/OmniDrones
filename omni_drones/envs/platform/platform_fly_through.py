from functorch import vmap
import torch
import torch.distributions as D

import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

import omni_drones.utils.kit as kit_utils
from pxr import UsdPhysics

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import cpos, off_diag, others
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.scene import design_scene
from omni_drones.utils.torch import euler_to_quaternion

from .utils import create_frame


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

def create_obstacle(
    prim_path: str,
    translation=(0., 0., 2.5),
    height: float=5,
):
    prim = prim_utils.create_prim(
        prim_path=prim_path,
        prim_type="Capsule",
        translation=translation,
        attributes={"radius":0.05, "height": height}
    )
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    kit_utils.set_collision_properties(
        prim_path, contact_offset=0.02, rest_offset=0
    )

    stage = prim_utils.get_current_stage()
    script_utils.createJoint(stage, "Fixed", prim.GetParent(), prim)
    return prim


class PlatformFlyThrough(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.reset_on_collision = self.cfg.task.reset_on_collision

        self.obstacles = RigidPrimView(
            "/World/envs/env_*/obstacle_*",
            reset_xform_properties=False,
            shape=[self.num_envs, -1],
            track_contact_forces=True
        )
        self.obstacles.initialize()

        self.drone.initialize(f"/World/envs/env_.*/platform/{self.drone.name}_*")
        self.init_drone_vels = torch.zeros_like(self.drone.get_velocities())

        self.frame_view = RigidPrimView(
            "/World/envs/env_.*/platform/frame",
            reset_xform_properties=False,
            track_contact_forces=True
        )
        self.frame_view.initialize()
        self.frame_view.post_reset()

        self.init_frame_poses = self.frame_view.get_world_poses(clone=True)
        self.init_frame_vels = torch.zeros_like(self.frame_view.get_velocities())
        # self.obstacle_pos = self.get_env_poses(self.obstacles.get_world_poses())[0]

        drone_state_dim = self.drone.state_spec.shape.numel()
        observation_spec = CompositeSpec({
            "state_self": UnboundedContinuousTensorSpec((1, drone_state_dim + self.drone.n)),
            "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, drone_state_dim)),
            "state_frame": UnboundedContinuousTensorSpec((1, 19)),
            # "obstacles": UnboundedContinuousTensorSpec((3, 2)),
        }).to(self.device)
        state_spec = CompositeSpec({
            "state_drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim + self.drone.n)),
            "state_frame": UnboundedContinuousTensorSpec((1, 19)),
            # "obstacles": UnboundedContinuousTensorSpec((3, 2)),
        }).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            4,
            observation_spec,
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
            state_spec
        )

        self.init_pos_dist = D.Uniform(
            torch.tensor([-2.0, -2., 1.5], device=self.device),
            torch.tensor([-1.5, 2.0, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2], device=self.device) * torch.pi
        )
        self.target_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.0], device=self.device),
            torch.tensor([0.2, 0.2, 2.0], device=self.device)
        )
        self.target_pos = torch.tensor([1.5, 0., 2.], device=self.device)
        self.target_heading =  torch.zeros(self.num_envs, 3, device=self.device)
        self.target_up = torch.zeros(self.num_envs, 3, device=self.device)

        self.alpha = 0.7
        info_spec = CompositeSpec({
            "pos_error": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["info"] = info_spec
        self.info = info_spec.zero()

    def _design_scene(self):
        drone_model = self.cfg.task.drone_model
        self.drone: MultirotorBase = MultirotorBase.REGISTRY[drone_model]()
        n = 4
        arm_length = self.cfg.task.arm_length
        self.drone_translations = torch.tensor([
            [-arm_length, 0.0, 0.0], 
            [0.0, arm_length, 0.0], 
            [arm_length, 0.0, 0.0], 
            [0.0, -arm_length, 0.0]
        ], device=self.device)
        self.drone_rotations = torch.tensor([1., 0., 0., 0.], device=self.device).expand(4, -1)

        arm_angles = [torch.pi * 2 / n * i for i in range(n)]
        arm_lengths = [arm_length for _ in range(n)]

        platform = prim_utils.create_prim(
            "/World/envs/env_0/platform", translation=(-1.5, 0., 2.)
        )
        self.drone.spawn(
            translations=self.drone_translations,
            prim_paths=[
                f"/World/envs/env_0/platform/{self.drone.name}_{i}" for i in range(n)
            ],
        )
        create_frame(
            "/World/envs/env_0/platform/frame",
            arm_angles,
            arm_lengths,
            [
                f"/World/envs/env_0/platform/{self.drone.name}_{i}/base_link"
                for i in range(n)
            ],
            enable_collision=True
        )
        design_scene()

        obstacle_spacing = self.cfg.task.obstacle_spacing
        create_obstacle("/World/envs/env_0/obstacle_0", translation=(0.,-obstacle_spacing, 2.5))
        create_obstacle("/World/envs/env_0/obstacle_1", translation=(0., 0., 2.5))
        create_obstacle("/World/envs/env_0/obstacle_2", translation=(0., obstacle_spacing, 2.5))
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        frame_pos = self.init_pos_dist.sample(env_ids.shape) + self.envs_positions[env_ids]
        frame_rpy = self.init_rpy_dist.sample(env_ids.shape)
        frame_rot = euler_to_quaternion(frame_rpy)
        self.frame_view.set_world_poses(frame_pos, frame_rot, env_indices=env_ids)
        self.frame_view.set_velocities(self.init_frame_vels[env_ids], env_ids)

        drone_pos, drone_rot = vmap(compose_transform)(
            frame_pos.unsqueeze(1),
            frame_rot.unsqueeze(1),
            translation=self.drone_translations,
            rotation=self.drone_rotations
        )
        self.drone.set_world_poses(drone_pos, drone_rot, env_ids)
        self.drone.set_velocities(self.init_drone_vels[env_ids], env_ids)

        self.info["pos_error"][env_ids] = 0
        self.info["collision"][env_ids] = 0


    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        drone_pos = self.drone_states[..., :3]
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)

        frame_pos, frame_rot = self.get_env_poses(self.frame_view.get_world_poses(clone=True))
        self.frame_heading = torch_utils.quat_axis(frame_rot, 0)
        self.frame_up = torch_utils.quat_axis(frame_rot, 2)
        self.frame_vels = self.frame_view.get_velocities(clone=True)

        self.target_frame_rpos = self.target_pos - frame_pos

        frame_drone_rpos = frame_pos.unsqueeze(1) - self.drone_states[..., :3]
        frame_state = torch.cat([
            self.target_frame_rpos, # 3
            frame_rot, # 4
            self.frame_heading, # 3
            self.frame_up, # 3
            self.frame_vels, # 6
        ], dim=-1).unsqueeze(1) # [num_envs, 1, 19]

        # obstacle_frame_rpos = self.obstacle_pos[..., [0, 1]] - frame_pos[:, [0, 1]].unsqueeze(1)

        identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)

        obs = TensorDict({}, [self.num_envs, self.drone.n])
        obs["state_self"] = torch.cat(
            [-frame_drone_rpos, self.drone_states[..., 3:], identity], dim=-1
        ).unsqueeze(2)
        obs["state_others"] = torch.cat(
            [self.drone_rpos, vmap(others)(self.drone_states[..., 3:])], dim=-1
        )
        obs["state_frame"] = frame_state.unsqueeze(1).expand(-1, self.drone.n, 1, -1)
        # obs["obstacles"] = obstacle_frame_rpos.unsqueeze(1).expand(-1, self.drone.n, 3, 2)

        state = TensorDict({}, [self.num_envs])
        state["state_drones"] = obs["state_self"].squeeze(2)    # [num_envs, drone.n, drone_state_dim]
        state["state_frame"] = frame_state                # [num_envs, 1, frame_state_dim]
        # state["obstacles"] = obstacle_frame_rpos    # [num_envs, 3, 2]
        
        pos_error = torch.norm(self.target_frame_rpos, dim=-1, keepdim=True)
        self.info["pos_error"].mul_(self.alpha).add_((1-self.alpha) * pos_error)
        return TensorDict(
            {
                "drone.obs": obs,
                "drone.state": state,
                "info": self.info
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        distance = torch.norm(self.target_frame_rpos, dim=-1, keepdim=True)
        
        reward = torch.zeros(self.num_envs, 4, 1, device=self.device)
        reward_pose = 1 / (1 + torch.square(distance * self.reward_distance_scale))
        
        spinnage = self.frame_vels[:, -3:].abs().sum(-1, keepdim=True)
        reward_spin = 1. / (1 + torch.square(spinnage))
        
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort).mean(-1, keepdim=True)

        reward[:] = (
            reward_pose 
            + reward_pose * (reward_spin) 
            + reward_effort
        ).unsqueeze(1)

        done_misbehave = (self.drone_states[..., 2] < 0.2).any(-1, keepdim=True)
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

