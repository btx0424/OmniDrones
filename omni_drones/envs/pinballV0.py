import functorch

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, normalize
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.objects as objects
import omni.isaac.core.materials as materials
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
)
from pxr import UsdShade


class PinballV0(IsaacEnv):
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
            shape=(-1, 1)
        )
        self.ball.initialize()

        self.init_ball_pos_dist = D.Uniform(
            torch.tensor([-1., -1., 2.5], device=self.device),
            torch.tensor([1., 1., 3.], device=self.device)
        )
        self.ball_mass_dist = D.Uniform(
            torch.tensor([0.01], device=self.device),
            torch.tensor([0.05], device=self.device)
        ) 

        self.init_drone_pos_dist = D.Uniform(
            torch.tensor([-1., -1., 1.25], device=self.device),
            torch.tensor([1., 1., 1.5], device=self.device)
        )
        self.init_drone_rpy_dist = D.Uniform(
            torch.tensor([-.1, -.1, 0.], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, 2.], device=self.device) * torch.pi
        )

        self.alpha = 0.8

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
            color=torch.tensor([1., .2, .2]),
            physics_material=material,
        )
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        drone_prims = self.drone.spawn(translations=[(0.0, 0.0, 2.)])
        material = UsdShade.Material(material.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("base_link/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
        
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "drone.obs": UnboundedContinuousTensorSpec((1, observation_dim)) ,
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "drone.action": self.drone.action_spec.unsqueeze(0),
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "drone.reward": UnboundedContinuousTensorSpec((1, 1))
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key="drone.obs",
            action_key="drone.action",
            reward_key="drone.reward",
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "score": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        
        drone_pos = self.init_drone_pos_dist.sample((*env_ids.shape, 1))
        drone_rpy = self.init_drone_rpy_dist.sample((*env_ids.shape, 1))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(torch.zeros(len(env_ids), 6, device=self.device), env_ids)

        # ball_pos = self.init_ball_pos_dist.sample((*env_ids.shape, 1))
        ball_pos = drone_pos + torch.tensor([0., 0., 1.6], device=self.device)
        ball_rot = torch.tensor([1., 0., 0., 0.], device=self.device).repeat(len(env_ids), 1)
        self.ball.set_world_poses(
            ball_pos + self.envs_positions[env_ids].unsqueeze(1), ball_rot, env_ids
        )
        self.ball.set_velocities(torch.zeros(len(env_ids), 6, device=self.device), env_ids)
        self.ball.set_masses(self.ball_mass_dist.sample(env_ids.shape), env_ids)
        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]
        self.ball_pos, ball_rot = self.get_env_poses(self.ball.get_world_poses())
        self.ball_vel = self.ball.get_velocities()

        # relative position and heading
        self.rpos =  self.ball_pos - self.drone.pos
        
        obs = [self.rpos, self.root_state[..., 3:], self.ball_vel[..., :3]]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        return TensorDict({
            "drone.obs": obs,
            "stats": self.stats,
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        
        # score = self.drone.base_link.get_net_contact_forces().any(-1).float()
        reward_pos = 1 / (1 + torch.norm(self.rpos[..., :2], dim=-1))
        reward_height = self.ball_pos[..., 2] - self.drone.pos[..., 2].clip(1.0)
        # reward_score = score * 3.
        reward = reward_pos + 0.8 * reward_height# + reward_score

        misbehave = (
            (self.drone.pos[..., 2] < 0.3) 
            | (self.ball_pos[..., 2] < 0.2)
            | (self.ball_pos[..., 2] > 4.)
            | (self.ball_pos[..., :2].abs() > 2.).any(-1)
        )
        
        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | misbehave
        )

        # self.stats["score"].add_(score)
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "reward": {"drone.reward": reward.unsqueeze(-1)},
                "done": done,
            },
            self.batch_size,
        )
