import torch
import functorch
from torchrl.data import UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict, TensorDictBase

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.prims import RigidPrimView


from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
import omni.isaac.core.utils.prims as prim_utils
import omni_drones.utils.scene as scene_utils

class Platform(IsaacEnv):
    
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize(f"/World/envs/env_.*/platform/{self.drone.name}_*")
        self.init_drone_poses = self.drone.get_env_poses(clone=True)
        self.init_drone_vels = torch.zeros_like(self.drone.get_velocities())
        
        self.frame_view = RigidPrimView("/World/envs/env_.*/platform/frame")
        self.frame_view.initialize()
        self.frame_view.post_reset()
        self.init_frame_poses = self.frame_view.get_world_poses(clone=True)
        self.init_frame_vels = torch.zeros_like(self.frame_view.get_velocities())

        self.agent_spec["drone"] = AgentSpec(
            "drone", 4, 
            UnboundedContinuousTensorSpec(35).to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )

        self.init_pos_scale = torch.tensor([2., 2., 0.6], device=self.device) 

    def _design_scene(self):
        drone_model = "Firefly"
        self.drone: MultirotorBase = MultirotorBase.REGISTRY[drone_model]()
        n = 4
        translations = torch.tensor([
            [-1., 0., 0.],
            [0., 1., 0.],
            [1., 0., 0.],
            [0., -1., 0.]
        ])
        arm_angles = [torch.pi*2/n * i for i in range(n)]
        arm_lengths = [1. for _ in range(n)]

        self.target_pos = torch.tensor([(0, 0, 1.5)], device=self.device)
        platform = prim_utils.create_prim(
            "/World/envs/env_0/platform",
            translation=self.target_pos
        )
        self.drone.spawn(
            translations=translations,
            prim_paths=[f"/World/envs/env_0/platform/{self.drone.name}_{i}" for i in range(n)]
        )
        scene_utils.create_frame(
            "/World/envs/env_0/platform/frame", 
            arm_angles, arm_lengths,
            [f"/World/envs/env_0/platform/{self.drone.name}_{i}/base_link" for i in range(n)]
        )
        scene_utils.design_scene()
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        offset = (
            torch.rand(len(env_ids), 3, device=self.device) 
            * self.init_pos_scale
            - self.init_pos_scale/2
        )

        pos, rot = self.init_frame_poses
        self.frame_view.set_world_poses(pos[env_ids]+offset, rot[env_ids], env_ids)
        self.frame_view.set_velocities(self.init_frame_vels[env_ids], env_ids)

        pos, rot = self.init_drone_poses
        self.drone.set_env_poses(pos[env_ids]+offset.unsqueeze(-2), rot[env_ids], env_ids)
        self.drone.set_velocities(self.init_drone_vels[env_ids], env_ids)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict["drone.action"]
        self.effort = self.drone.apply_action(actions)
    
    def _compute_state_and_obs(self):
        drone_state = self.drone.get_state()
        frame_pos, frame_rot = self.frame_view.get_world_poses(clone=True)
        frame_pos = frame_pos - self.envs_positions
        
        target_rel_pos = self.target_pos - frame_pos
        
        drone_rel_state = torch.cat([
            self.target_pos-drone_state[..., :3],
            drone_state[..., 3:],
        ], dim=-1)

        obs = torch.cat([
            drone_rel_state,
            frame_pos.unsqueeze(1)-drone_state[..., :3],
            frame_rot.unsqueeze(1).expand(self.num_envs, 4, 4),
            target_rel_pos.unsqueeze(1).expand(self.num_envs, 4, 3)
        ], dim=-1)

        state = torch.cat([
            drone_rel_state.flatten(-2),
            target_rel_pos,
            frame_rot
        ], dim=-1)

        return TensorDict({
            "drone.obs": obs,
            "drone.state": state,
        }, self.batch_size)
    
    def _compute_reward_and_done(self):
        frame_pos, frame_rot = self.frame_view.get_world_poses(clone=True)
        frame_pos = frame_pos - self.envs_positions

        linvel, angvel = self.frame_view.get_velocities().split([3, 3], dim=-1)

        distance = torch.norm(frame_pos - self.target_pos, dim=-1, keepdim=True)

        reward = torch.zeros(self.num_envs, 4, 1, device=self.device)
        reward_pos = 1 / (1 + torch.square(distance))
        reward_rot = 1 / (1 + torch.square(angvel).sum(-1, keepdim=True))

        reward[:] = (reward_pos + reward_pos * reward_rot).unsqueeze(1)

        self._tensordict["return"] += reward

        misbehave = (
            (frame_pos[..., 2] < 0.1).unsqueeze(-1)
            | (distance > 5.)
        )

        done = (
            (self.progress_buf >= self.max_eposode_length).unsqueeze(-1)
            | (misbehave & (self.progress_buf >= self.min_episode_length).unsqueeze(-1))
        )

        return TensorDict({
            "reward": {
                "drone.reward": reward
            },
            "return": self._tensordict["return"],
            "done": done
        }, self.batch_size)

