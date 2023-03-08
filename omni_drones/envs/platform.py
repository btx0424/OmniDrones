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
        self.drone.initialize("/World/envs/env_.*/platform/Firefly_*")
        self.frame_view = RigidPrimView("/World/envs/env_.*/platform/frame")
        self.frame_view.initialize()

        self.init_poses = self.drone.get_env_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.agent_spec["drone"] = AgentSpec(
            "drone", 4, 
            self.drone.state_spec.to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )

        self.init_pos_scale = torch.tensor([2., 2., 0.6], device=self.device) 
        self.init_pos_offset = torch.tensor([-1., -1., 0.4], device=self.device)

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
        platform = prim_utils.create_prim(
            "/World/envs/env_0/platform",
            translation=(0, 0, 1.5)
        )
        self.drone.spawn(
            translations=translations,
            prim_paths=[f"/World/envs/env_0/platform/Firefly_{i}" for i in range(n)]
        )
        scene_utils.create_frame(
            "/World/envs/env_0/platform/frame", 
            arm_angles, arm_lengths,
            [f"/World/envs/env_0/platform/Firefly_{i}/base_link" for i in range(n)]
        )
        scene_utils.design_scene()
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        pos, rot = self.init_poses
        self.drone._reset_idx(env_ids)
        self.drone.set_env_poses(pos[env_ids], rot[env_ids], env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict["drone.action"]
        self.effort = self.drone.apply_action(actions)
    
    def _compute_state_and_obs(self):
        drone_state = self.drone.get_state()
        platform_state = self.frame_view.get_local_poses()
        
        return TensorDict({
            "drone.obs": drone_state,
            # "platform.state": 
        }, self.batch_size)
    
    def _compute_reward_and_done(self):
        reward = torch.zeros(self.num_envs, 4, device=self.device)
        self._tensordict["drone.return"] += reward.unsqueeze(-1)
        done  = (
            (self.progress_buf >= self.max_eposode_length).unsqueeze(-1)
        )
        return TensorDict({
            "reward": {
                "drone.reward": reward.unsqueeze(-1)
            },
            "done": done
        }, self.batch_size)