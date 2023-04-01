import functorch
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.core.utils.torch as torch_utils
from omni_drones.utils.torch import euler_to_quaternion
import torch
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.prims import RigidPrimView
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec

import omni_drones.utils.scene as scene_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.debug_draw import _debug_draw

class Platform(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_effort_weight = self.cfg.task.get("reward_effort_weight", 0.1)
        self.reward_distance_scale = self.cfg.task.get("reward_distance_scale", 1.)

        self.drone.initialize(f"/World/envs/env_.*/platform/{self.drone.name}_*")
        self.target.initialize()
        self.init_drone_poses = self.drone.get_world_poses(clone=True)
        self.init_drone_vels = torch.zeros_like(self.drone.get_velocities())

        self.frame_view = RigidPrimView("/World/envs/env_.*/platform/frame")
        self.frame_view.initialize()
        self.frame_view.post_reset()
        self.init_frame_poses = self.frame_view.get_world_poses(clone=True)
        self.init_frame_vels = torch.zeros_like(self.frame_view.get_velocities())

        drone_state_dim = self.drone.state_spec.shape.numel()
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            4,
            UnboundedContinuousTensorSpec(drone_state_dim+12 + self.drone.n).to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
            state_spec=UnboundedContinuousTensorSpec(drone_state_dim*self.drone.n+9).to(self.device)
        )

        self.init_pos_scale = torch.tensor([4.0, 4.0, 1.0], device=self.device)
        self.init_rpy_scale = self.init_rpy_scale = torch.tensor([0.6, 0.6, 2.0], device=self.device) * torch.pi
        self.target_pos = torch.tensor([0., 0., 2.], device=self.device)
        self.target_heading =  torch.zeros(self.num_envs, 3, device=self.device)

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        drone_model = self.cfg.task.drone_model
        self.drone: MultirotorBase = MultirotorBase.REGISTRY[drone_model]()
        n = 4
        translations = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        arm_angles = [torch.pi * 2 / n * i for i in range(n)]
        arm_lengths = [1.0 for _ in range(n)]

        self.target = VisualSphere(
            prim_path="/World/envs/env_0/target",
            name="target",
            translation=(0., 0., 2.),
            radius=0.05,
            color=torch.tensor([1.0, 0.0, 0.0]),
        )

        platform = prim_utils.create_prim(
            "/World/envs/env_0/platform", translation=(0., 0., 2.)
        )
        self.drone.spawn(
            translations=translations,
            prim_paths=[
                f"/World/envs/env_0/platform/{self.drone.name}_{i}" for i in range(n)
            ],
        )
        scene_utils.create_frame(
            "/World/envs/env_0/platform/frame",
            arm_angles,
            arm_lengths,
            [
                f"/World/envs/env_0/platform/{self.drone.name}_{i}/base_link"
                for i in range(n)
            ],
        )
        scene_utils.design_scene()
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        offset = (
            torch.rand(len(env_ids), 3, device=self.device) * self.init_pos_scale
            - self.init_pos_scale / 2
        )

        pos, rot = self.init_frame_poses
        new_poses = (pos[env_ids] + offset, rot[env_ids])
        self.frame_view.set_world_poses(*new_poses, env_ids)
        self.frame_view.set_velocities(self.init_frame_vels[env_ids], env_ids)

        pos, rot = self.init_drone_poses
        new_poses = (pos[env_ids] + offset.unsqueeze(-2), rot[env_ids])
        self.drone.set_world_poses(*new_poses, env_ids)
        self.drone.set_velocities(self.init_drone_vels[env_ids], env_ids)

        target_rpy = torch.zeros(len(env_ids), 3, device=self.device)
        target_rpy[..., 2] = torch.rand(len(env_ids), device=self.device) * torch.pi - torch.pi / 2
        target_rot = euler_to_quaternion(target_rpy)
        self.target_heading[env_ids] = torch_utils.quat_axis(target_rot, 0)


    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()

        frame_pos, frame_rot = self.get_env_poses(self.frame_view.get_world_poses(clone=True))
        frame_heading = torch_utils.quat_axis(frame_rot, 0)
        self.frame_up = torch_utils.quat_axis(frame_rot, 1)

        self.target_frame_rpos = self.target_pos - frame_pos
        self.target_frame_rheading = self.target_frame_rpos + self.target_heading - frame_heading

        target_drone_rpos = self.target_pos - self.drone_states[..., :3]
        frame_drone_rpos = frame_pos.unsqueeze(1) - self.drone_states[..., :3]

        identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)
        obs = torch.cat(
            [
                frame_drone_rpos, # 3
                target_drone_rpos, # 3
                self.drone_states[..., 3:], # drone_state_dim - 3
                self.target_frame_rpos.unsqueeze(1).expand(-1, self.drone.n, -1), # 3
                self.target_frame_rheading.unsqueeze(1).expand(-1, self.drone.n, -1), # 3
                self.frame_up.unsqueeze(1).expand(-1, self.drone.n, -1), # 3
                identity
            ],
            dim=-1,
        )

        state = torch.cat(
            [
                frame_drone_rpos.flatten(1), # 3 * drone.n
                self.drone_states[..., 3:].flatten(1), # (drone_state_dim - 3) * drone.n
                self.target_frame_rpos, # 3
                self.target_frame_rheading, # 3
                self.frame_up # 3
            ], dim=-1
        )

        if self._should_render(0):
            env_pos = self.envs_positions[self.central_env_idx]
            _frame_pos = frame_pos[self.central_env_idx] + env_pos
            _target_pos = self.target_pos + env_pos
            point_list_0 = []
            point_list_1 = []
            point_list_0.append(_frame_pos.tolist())
            point_list_0.append((_frame_pos + frame_heading[self.central_env_idx]).tolist())
            point_list_1.append(_target_pos.tolist())
            point_list_1.append((_target_pos + self.target_heading[self.central_env_idx]).tolist())
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [1 for _ in range(len(point_list_0))]

            self.draw.clear_lines()
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)
            
        return TensorDict(
            {
                "drone.obs": obs,
                "drone.state": state,
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        vels = self.frame_view.get_velocities()
    
        distance = torch.norm(
            torch.cat([self.target_frame_rpos, self.target_frame_rheading], dim=-1)
        , dim=-1, keepdim=True)

        reward = torch.zeros(self.num_envs, 4, 1, device=self.device)
        reward_pos = 1 / (1 + torch.square(distance * self.reward_distance_scale))
        reward_rot = 1 / (1 + torch.square(vels[..., -1].unsqueeze(-1)))
        reward_up = 1 / (1 + torch.square(1-self.frame_up[:, 2]).unsqueeze(1))
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort).mean(-1, keepdim=True)

        reward[:] = (reward_pos + reward_pos * (reward_rot + reward_up) + reward_effort).unsqueeze(1)

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

class PlatformTracking(Platform):
    def _compute_state_and_obs(self):
        return super()._compute_state_and_obs()
    
    def _compute_reward_and_done(self):
        return super()._compute_reward_and_done()

