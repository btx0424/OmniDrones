import functorch
import numpy as np

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
import torch
from omni.isaac.core.objects import (
    DynamicSphere,
    FixedCuboid,
    FixedCylinder,
    VisualSphere,
)
from omni.isaac.core.prims import GeometryPrimView, RigidPrimView
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

# drones on land by default
# only cubes are available as walls
# clip state as walls


class Prey(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.obstacle = GeometryPrimView(prim_paths_expr="/World/envs/.*/obstacle")
        self.target = RigidPrimView(prim_paths_expr="/World/envs/.*/target")
        self.obstacle.initialize()
        self.target.initialize()
        self.target.post_reset()
        self.target_init_vel = self.target.get_velocities(clone=True)
        self.target_init_vel[..., 0] = 1.0
        self.env_ids = torch.from_numpy(np.arange(0, cfg.env.num_envs))
        self.env_width = cfg.env.env_spacing / 2.0

        self.init_poses = self.drone.get_env_poses(clone=True)

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.num_agents,
            UnboundedContinuousTensorSpec(26, device=self.device),
            UnboundedContinuousTensorSpec(7, device=self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )

        self.vels = self.drone.get_velocities()
        self.init_pos_scale = torch.tensor([2.0, 2.0, 0.6], device=self.device)
        self.init_pos_offset = torch.tensor([-1.0, -1.0, 0.3], device=self.device)

    def _design_scene(self):
        self.num_agents = 1
        cfg = RobotCfg()
        cfg.rigid_props.max_linear_velocity = 1.0
        self.drone: MultirotorBase = MultirotorBase.REGISTRY["Crazyflie"](cfg=cfg)
        # self.drone: MultirotorBase = MultirotorBase.REGISTRY["Firefly"](cfg=cfg)

        self.target_pos = torch.tensor([[0.0, 0.0, 1.5]], device=self.device)
        self.obstacle_pos = torch.tensor([[1.0, 0.5, 0]], device=self.device)
        DynamicSphere(
            prim_path="/World/envs/env_0/target",
            name="target",
            translation=self.target_pos,
            radius=0.05,
            color=torch.tensor([1.0, 0.0, 0.0]),
        )

        FixedCylinder(
            prim_path="/World/envs/env_0/obstacle",
            name="obstacle",
            translation=self.obstacle_pos,
            # orientation=torch.tensor([1., 0., 0., 0.]),
            color=torch.tensor([0.1, 0.0, 0.0]),
            radius=0.3,
            height=5,
        )

        kit_utils.set_nested_rigid_body_properties(
            prim_path="/World/envs/env_0/target",
            max_linear_velocity=0.7,
        )

        kit_utils.set_rigid_body_properties(
            prim_path="/World/envs/env_0/target", disable_gravity=True
        )
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        n = self.num_agents
        for i in range(n):
            translation = torch.zeros(n, 3)
            translation[:, 0] = i
            translation[:, 1] = torch.arange(n)
            translation[:, 2] = 0.5
        self.drone.spawn(n, translation)  # to make n drones
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        n = self.num_agents
        _, rot = self.init_poses
        self.drone._reset_idx(env_ids)
        pos = (
            torch.rand(len(env_ids), n, 3, device=self.device) * self.init_pos_scale
            + self.init_pos_offset
        )
        self.drone.set_env_poses(pos, rot[env_ids], env_ids)
        self.drone.set_velocities(torch.zeros_like(self.vels[env_ids]), env_ids)

        self.target.set_world_poses(
            (self.envs_positions + self.target_pos)[env_ids], indices=env_ids
        )
        self.target.set_velocities(self.target_init_vel[env_ids], indices=env_ids)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict["drone.action"]
        self.effort = self.drone.apply_action(actions)
        # self.drone.apply_action(self._get_dummy_policy_drone())
        self.target.apply_forces(self._get_dummy_policy_prey())

        # restriction on (x,y)
        agent_pos, agent_rot = self.drone.get_env_poses()
        target_pos, target_rot = self.target.get_world_poses()
        target_pos -= self.envs_positions
        agent_pos[..., :2] = agent_pos[..., :2].clamp(-self.env_width, self.env_width)
        target_pos[..., :2] = target_pos[..., :2].clamp(-self.env_width, self.env_width)
        self.drone.set_env_poses(agent_pos, agent_rot, self.env_ids)
        self.target.set_world_poses(
            target_pos + self.envs_positions, target_rot, self.env_ids
        )

    def _compute_state_and_obs(self):
        drone_state = self.drone.get_state()
        prey_state = self.target.get_world_poses()[
            0
        ] - self.drone._envs_positions.squeeze(1)
        prey_pos = prey_state.unsqueeze(1).expand(-1, self.num_agents, -1)
        obs = torch.cat((drone_state, prey_pos), -1)

        return TensorDict({"drone.obs": obs}, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot = self.drone.get_env_poses(False)
        prey_state = self.target.get_world_poses()[
            0
        ] - self.drone._envs_positions.squeeze(1)
        prey_pos = prey_state.unsqueeze(1).expand(-1, self.num_agents, -1)
        target_dist = torch.norm(pos - prey_pos, dim=-1)
        catch_reward = (target_dist < 0.2) * 1.0
        reward = catch_reward - target_dist
        self._tensordict["drone.return"] += reward.unsqueeze(-1)
        done = (self.progress_buf >= self.max_eposode_length).unsqueeze(-1) | (
            catch_reward > 0
        )
        # done should be integrated
        return TensorDict(
            {"reward": {"drone.reward": reward.unsqueeze(-1)}, "done": done.any(-1)},
            self.batch_size,
        )

    def _get_dummy_policy_prey(self):
        pos, rot = self.drone.get_env_poses(False)
        prey_state = self.target.get_world_poses()[
            0
        ] - self.drone._envs_positions.squeeze(1)
        prey_pos = prey_state.unsqueeze(1).expand(-1, self.num_agents, -1)
        dist_pos = torch.norm(prey_pos - pos, dim=-1).unsqueeze(1).expand(-1, -1, 3)
        orient = (prey_pos - pos) / dist_pos
        force = (orient * 1.0 / dist_pos).sum(-2)
        return force

    def _get_dummy_policy_drone(self):
        pos, rot = self.drone.get_env_poses(False)
        prey_state = self.target.get_world_poses()[
            0
        ] - self.drone._envs_positions.squeeze(1)
        prey_pos = prey_state.unsqueeze(1).expand(-1, self.num_agents, -1)
        dist_pos = torch.norm(prey_pos - pos, dim=-1).unsqueeze(1).expand(-1, -1, 3)
        _vel = (prey_pos - pos) / dist_pos
        _pos = _vel + pos
        return _pos, _vel
