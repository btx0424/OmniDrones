import torch
from functorch import vmap
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, BinaryDiscreteTensorSpec

import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import omni.isaac.core.objects as objects
from pxr import UsdPhysics

import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.utils.torch import cpos, off_diag, others, make_cells
from omni_drones.robots.drone import MultirotorBase
from omni_drones.sensors.camera import Camera, PinholeCameraCfg
from omni_drones.views import ArticulationView, RigidPrimView

from omni_drones.robots import ASSET_PATH

def sample_from_grid(cells: torch.Tensor, n):
    idx = torch.randperm(cells.shape[0], device=cells.device)[:n]
    return cells[idx]

class MultiGate(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.reset_on_collision = self.cfg.task.reset_on_collision
        self.gate_moving_range = self.cfg.task.gate_moving_range
        self.gate_scale = self.cfg.task.gate_scale
        self.time_encoding = self.cfg.task.time_encoding
        self.safe_distance = self.cfg.task.safe_distance

        self.drone.initialize()
        
        self.gate = ArticulationView(
            "/World/envs/env_*/Gate",
            reset_xform_properties=False,
            shape=[self.num_envs, 1],
        )
        self.gate.initialize()
        self.gate_frame = RigidPrimView(
            "/World/envs/env_*/Gate/frame",
            reset_xform_properties=False,
            shape=[self.num_envs, 1],
            track_contact_forces=True
        )
        self.gate_frame.initialize()

        self.target = RigidPrimView(
            "/World/envs/env_*/target",
            reset_xform_properties=False,
        )
        self.target.initialize()

        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())
        self.crossed_plane = torch.zeros(self.num_envs, self.drone.n, device=self.device, dtype=bool)

        self.target_pos = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)
        self.target_pos[..., 0] = 1.5
        self.target_pos[..., 1] = torch.linspace(-1.2, 1.2, self.drone.n)
        self.target_pos[..., 2] = 1.5

        obs_dim = self.drone.state_spec.shape[-1]
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_dim += self.time_encoding_dim
        
        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, obs_dim + 6)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, 13+1)),
        }).to(self.device)
        
        state_spec = CompositeSpec({
            'drones': UnboundedContinuousTensorSpec((self.drone.n, obs_dim + 6))
        })

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.drone.n,
            observation_spec.to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
            state_spec.to(self.device),
        )

        self.cells = make_cells(
            [-2.25, -1.2, 1.], 
            [-1.75, 1.2, 2.2], 
            [0.4, 0.4, 0.4],
        ).to(self.device).flatten(0, -2)
        
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([.2, .2, 0.], device=self.device) * torch.pi
        )
        self.init_gate_pos_dist = D.Uniform(
            torch.tensor([-self.gate_moving_range], device=self.device),
            torch.tensor([self.gate_moving_range], device=self.device)
        )
        
        self.alpha = 0.7

        stats_spec = CompositeSpec({
            "pos_error": UnboundedContinuousTensorSpec(self.drone.n),
            "collision": UnboundedContinuousTensorSpec(1),
            "success": BinaryDiscreteTensorSpec(self.drone.n, dtype=bool)
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

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
        
        translations = torch.zeros(4, 3)
        drone_prims = self.drone.spawn(translations=translations)
        scale = torch.ones(3) * self.cfg.task.gate_scale
        prim_utils.create_prim(
            "/World/envs/env_0/Gate",
            usd_path=ASSET_PATH + "/usd/gate_sliding.usd",
            translation=(0., 0., 2.0),
            scale=scale
        )
        target = objects.DynamicSphere(
            "/World/envs/env_0/target",
            translation=(1.5, 0., 2.),
            radius=0.05,
            color=torch.tensor([1., 0., 0.])
        )
        kit_utils.set_collision_properties(target.prim_path, collision_enabled=False)
        kit_utils.set_rigid_body_properties(target.prim_path, disable_gravity=True)
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        
        drone_pos = vmap(sample_from_grid, randomness="different")(
            self.cells.expand(len(env_ids), *self.cells.shape), n=self.drone.n
        )
        drone_rpy = self.init_rpy_dist.sample((*env_ids.shape, self.drone.n))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.drone.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.drone.set_joint_velocities(self.init_joint_vels[env_ids], env_ids)
        self.crossed_plane[env_ids] = False

        # target_pos = self.target_pos_dist.sample((*env_ids.shape, 1))
        # self.target_pos[env_ids] = target_pos
        # self.target.set_world_poses(
        #     target_pos + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids
        # )
        gate_pos = self.init_gate_pos_dist.sample(env_ids.shape)
        self.gate.set_joint_positions(
            gate_pos, env_indices=env_ids
        )
        self.gate.set_joint_velocities(
            torch.zeros_like(gate_pos), env_indices=env_ids
        )
        self.gate.set_joint_position_targets(
            torch.zeros_like(gate_pos), env_indices=env_ids
        )

        self.stats.exclude("success")[env_ids] = 0.
        self.stats["success"][env_ids] = False

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.info["drone_state"][:] = self.drone_state[..., :13]
        
        self.gate_pos, _ = self.get_env_poses(self.gate.get_world_poses())
        self.gate_vel = self.gate.get_velocities()

        # relative position
        target_drone_rpos = self.target_pos - self.drone_state[..., :3]
        gate_drone_rpos = self.gate_pos - self.drone_state[..., :3]
        
        drone_rpos = vmap(cpos)(self.drone.pos, self.drone.pos)
        drone_rpos = vmap(off_diag)(drone_rpos)
        self.drone_pdist = torch.norm(drone_rpos, dim=-1, keepdim=True)

        obs_self = [target_drone_rpos, gate_drone_rpos, self.drone_state]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs_self.append(t.expand(-1, self.drone.n, self.time_encoding_dim))
        
        obs_others = [drone_rpos, self.drone_pdist, vmap(others)(self.drone_state[..., 3:13])]
        
        obs = TensorDict({
            "obs_self": torch.cat(obs_self, dim=-1).unsqueeze(2),
            "obs_others": torch.cat(obs_others, dim=-1)
        }, [self.num_envs, self.drone.n])
        
        state = TensorDict({
            "drones": obs["obs_self"].squeeze(2)
        }, [self.num_envs])
        
        self.distance_to_target = torch.norm(target_drone_rpos, dim=-1)
        self.stats["pos_error"].lerp_(self.distance_to_target, (1-self.alpha))
        
        return TensorDict({
            "drone.obs": obs,
            "drone.state": state,
            "stats": self.stats,
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):        
        crossed_plane = self.drone.pos[..., 0] > 0.
        crossing_plane = (crossed_plane & (~self.crossed_plane))
        self.crossed_plane |= crossed_plane
        distance_to_gate_plane = 0. - self.drone.pos[..., 0]
        distance_to_gate_center = torch.abs(self.drone.pos[..., 1:] - self.gate_pos[..., 1:])
        through_gate = (distance_to_gate_center < 0.5).all(-1)

        gate_reward = torch.where(
            distance_to_gate_plane > 0.,
            (0.4 - distance_to_gate_center).sum(-1) * torch.exp(-distance_to_gate_plane),
            1.
        )

        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values
        reward_separation = torch.square(separation / self.safe_distance).clamp(0, 1)
        # pose_reward = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance_to_target))
        reward_pos = torch.exp(-self.reward_distance_scale * self.distance_to_target)
        
        # uprightness
        reward_up = 0.5 * torch.square((self.drone.up[..., 2] + 1) / 2)
        spin = torch.square(self.drone.vel[..., -1])
        reward_spin = 1. / (1.0 + torch.square(spin))

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)

        assert reward_pos.shape == reward_spin.shape
        reward = (
            reward_pos * 1.4
            + gate_reward
            + (reward_pos + 0.3) * (reward_up + reward_spin) 
            + reward_effort
        )
        
        done_invalid = (crossing_plane & ~through_gate)
        done_misbehave: torch.Tensor = (
            (self.drone.pos[..., 2] < 0.2) 
            | (self.distance_to_target > 6.)
        )

        collision = (
            self.gate_frame
            .get_net_contact_forces()
            .any(-1)
            .any(-1, keepdim=True)
        )
        self.stats["collision"].add_(collision.float())

        if self.reset_on_collision:
            done_misbehave.bitwise_or_(collision)
        
        # done_hasnan = torch.isnan(self.drone_state).any(-1)
        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | done_misbehave.any(-1, True)
            | done_invalid.any(-1, True)
            # | (separation < 0.25)
        )
        # self.stats["success"].bitwise_or_(self.distance_to_target < 0.2)

        self._tensordict["return"] += reward.unsqueeze(-1)
        return TensorDict(
            {
                "reward": {"drone.reward": reward.unsqueeze(-1)},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )
