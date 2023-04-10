import functorch
import torch
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import omni.isaac.core.objects as objects
from pxr import UsdPhysics

import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView

def create_obstacles(
    prim_path: str,
    translation=(0., 0., 1.5),
    width: float=5,
):
    prim = prim_utils.create_prim(
        prim_path=prim_path,
        prim_type="Capsule",
        translation=translation,
        attributes={"axis":"Y", "radius":0.05, "height": width}
    )
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    kit_utils.set_collision_properties(
        prim_path, contact_offset=0.02, rest_offset=0
    )

    stage = prim_utils.get_current_stage()
    script_utils.createJoint(stage, "Fixed", prim.GetParent(), prim)
    return prim

def create_payload(
    drone_prim_path: str,
    bar_length: str,
    payload_radius: float=0.08,
    payload_mass: float=0.3
):
    bar = prim_utils.create_prim(
        prim_path=drone_prim_path + "/bar",
        prim_type="Capsule",
        translation=(0., 0., -bar_length / 2.),
        attributes={"radius": 0.01, "height": bar_length}
    )
    bar.GetAttribute('primvars:displayColor').Set([(0.8, 0.1, 0.1)])
    UsdPhysics.RigidBodyAPI.Apply(bar)
    UsdPhysics.CollisionAPI.Apply(bar)
    massAPI = UsdPhysics.MassAPI.Apply(bar)
    massAPI.CreateMassAttr().Set(0.02)

    base_link = prim_utils.get_prim_at_path(drone_prim_path + "/base_link")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "D6", bar, base_link)
    joint.GetAttribute("limit:rotX:physics:low").Set(-120)
    joint.GetAttribute("limit:rotX:physics:high").Set(120)
    joint.GetAttribute("limit:rotY:physics:low").Set(-120)
    joint.GetAttribute("limit:rotY:physics:high").Set(120)
    UsdPhysics.DriveAPI.Apply(joint, "rotX")
    UsdPhysics.DriveAPI.Apply(joint, "rotY")
    joint.GetAttribute("drive:rotX:physics:damping").Set(0.0001)
    joint.GetAttribute("drive:rotY:physics:damping").Set(0.0001)
    # joint.GetAttribute('physics:excludeFromArticulation').Set(True)

    payload = objects.DynamicSphere(
        prim_path=drone_prim_path + "/payload",
        translation=(0., 0., -bar_length),
        radius=payload_radius,
        mass=payload_mass
    )
    joint = script_utils.createJoint(stage, "Fixed", bar, payload.prim)
    kit_utils.set_collision_properties(
        drone_prim_path + "/bar", contact_offset=0.02, rest_offset=0
    )
    kit_utils.set_collision_properties(
        drone_prim_path + "/payload", contact_offset=0.02, rest_offset=0
    )


class FlyThrough(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.reset_on_collision = self.cfg.task.reset_on_collision

        # self.bar = RigidPrimView(
        #     f"/World/envs/env_*/{self.drone.name}_*/bar",
        #     reset_xform_properties=False,
        #     track_contact_forces=True,
        #     contact_filter_prim_paths_expr=[
        #         "/World/envs/env_*/obstacle_0",
        #         "/World/envs/env_*/obstacle_1"
        #     ]
        # )

        # self.bar.initialize()
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
            reset_xform_properties=False,
        )
        self.payload.initialize()

        self.payload_target_vis = RigidPrimView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.payload_target_vis.initialize()

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())
        self.obstacle_pos = self.get_env_poses(self.obstacles.get_world_poses())[0]

        drone_state_dim = self.drone.state_spec.shape[-1]
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            UnboundedContinuousTensorSpec(drone_state_dim + 9 + 4).to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )

        self.init_pos_dist = D.Uniform(
            torch.tensor([-2.5, 0., 1.5], device=self.device),
            torch.tensor([-1.0, 0., 2.0], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([.2, .2, 2], device=self.device) * torch.pi
        )
        self.payload_target_pos_dist = D.Uniform(
            torch.tensor([1.3, 0., 1.0], device=self.device),
            torch.tensor([1.5, 0., 1.5], device=self.device)
        )
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(self.cfg.task.payload_mass_min, device=self.device),
            torch.as_tensor(self.cfg.task.payload_mass_max, device=self.device)
        )

        self.payload_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.alpha = 0.7

        info_spec = CompositeSpec({
            "payload_pos_error": UnboundedContinuousTensorSpec(1),
            "drone_uprightness": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["info"] = info_spec
        # self.info = info_spec.zero()
        self.payload_pos_error = torch.zeros(self.num_envs, 1, device=self.device)
        self.drone_uprightness = torch.zeros(self.num_envs, 1, device=self.device)
        self.collision = torch.zeros(self.num_envs, 1, device=self.device)
        self.info = TensorDict({
            "payload_pos_error": self.payload_pos_error,
            "drone_uprightness": self.drone_uprightness,
            "collision": self.collision
        }, self.num_envs)

    def _design_scene(self):
        cfg = RobotCfg()
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        
        obstacle_spacing = self.cfg.task.obstacle_spacing
        create_obstacles(
            "/World/envs/env_0/obstacle_0", 
            translation=(0., 0., 1.2)
        )
        create_obstacles(
            "/World/envs/env_0/obstacle_1", 
            translation=(0., 0., 1.2+obstacle_spacing)
        )

        self.drone.spawn(translations=[(0.0, 0.0, 2.)])
        create_payload(f"/World/envs/env_0/{self.drone.name}_0", self.cfg.task.bar_length)

        sphere = objects.DynamicSphere(
            "/World/envs/env_0/target",
            translation=(1.5, 0., 1.),
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

        payload_target_pos = self.payload_target_pos_dist.sample(env_ids.shape)
        self.payload_target_pos[env_ids] = payload_target_pos
        self.payload_target_vis.set_world_poses(
            payload_target_pos + self.envs_positions[env_ids], 
            env_indices=env_ids
        )
        payload_mass = self.payload_mass_dist.sample(env_ids.shape)
        self.payload.set_masses(payload_mass, env_ids)

        self.info["payload_pos_error"][env_ids] = 0
        self.info["drone_uprightness"][env_ids] = 0
        self.info["collision"][env_ids] = 0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.drone_up = self.drone_state[..., 16:19]
        self.payload_pos, payload_rot = self.get_env_poses(self.payload.get_world_poses())
        self.payload_vels = self.payload.get_velocities()

        # relative position and heading
        self.drone_payload_rpos = self.drone_state[..., :3] - self.payload_target_pos.unsqueeze(1)
        self.target_payload_rpos = (self.payload_target_pos - self.payload_pos).unsqueeze(1)
        obstacle_drone_rpos = self.obstacle_pos[..., [0, 2]] - self.drone_state[..., [0, 2]]
        
        obs = torch.cat([
            self.drone_payload_rpos,
            self.drone_state[..., 3:],
            self.target_payload_rpos, # 3
            self.payload_vels.unsqueeze(1), # 6
            obstacle_drone_rpos.flatten(start_dim=-2).unsqueeze(1),
        ], dim=-1)

        payload_pos_error = torch.norm(self.target_payload_rpos, dim=-1)
        self.info["payload_pos_error"].mul_(self.alpha).add_((1-self.alpha) * payload_pos_error)
        self.info["drone_uprightness"].mul_(self.alpha).add_((1-self.alpha) * self.drone_up[..., 2])
        
        return TensorDict({
            "drone.obs": obs,
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot, vels = self.drone_state[..., :13].split([3, 4, 6], dim=-1)
        
        # pose reward
        distance = torch.norm(self.target_payload_rpos, dim=-1)

        pose_reward = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
        # pose_reward = torch.exp(-distance * self.reward_distance_scale)
        # uprightness
        up_reward = torch.square((self.drone_up[..., 2] + 1) / 2)

        effort_reward = self.reward_effort_weight * torch.exp(-self.effort)

        spin = torch.square(vels[..., -1])
        spin_reward = 0.5 / (1.0 + torch.square(spin))

        swing = torch.norm(self.payload_vels[..., :3], dim=-1, keepdim=True)
        swing_reward = 0.5 * torch.exp(-swing)

        # collision = (
        #     self.bar
        #     .get_net_contact_forces()
        #     .any(-1, keepdim=True)
        # )
        collision = (
            self.obstacles
            .get_net_contact_forces()
            .any(-1)
            .any(-1, keepdim=True)
        )
        collision_reward = collision.float()

        self.info["collision"] += collision_reward
        assert pose_reward.shape == up_reward.shape == spin_reward.shape == swing_reward.shape
        reward = (
            pose_reward 
            + pose_reward * (up_reward + spin_reward + swing_reward) 
            + effort_reward
        )
        
        done_misbehave = (
            (pos[..., 2] < 0.2) 
            | (pos[..., 2] > 2.5)
            | (self.payload_pos[..., 2] < 0.15).unsqueeze(1)
        )
        done_hasnan = torch.isnan(self.drone_state).any(-1)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | done_misbehave
            | done_hasnan
        )
        
        if self.reset_on_collision:
            done = done | collision

        self._tensordict["return"] += reward.unsqueeze(-1)
        return TensorDict(
            {
                "reward": {"drone.reward": reward.unsqueeze(-1)},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )
