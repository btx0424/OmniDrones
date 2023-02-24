import torch
import functorch
from torchrl.data import UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict, TensorDictBase

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.objects import VisualSphere

from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec, Optional, List
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import (
    Crazyflie, Firefly, Neo11, Hummingbird
)
import omni_drones.utils.kit as kit_utils

class Formation(IsaacEnv):

    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.init_poses = self.drone.get_env_poses(clone=True)
        observaton_spec = ...
        self.agent_spec["drone"] = AgentSpec(
            "drone", 5,
            
            self.drone.action_spec,
            UnboundedContinuousTensorSpec(1).to(self.device)
        )
    
    def _design_scene(self) -> Optional[List[str]]:
        cfg = RobotCfg()
        self.drone = Firefly(cfg=cfg)
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.drone.spawn(translation=(0., 0., 1.))
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        _, rot = self.init_poses
        self.drone._reset_idx(env_ids)
        pos = torch.rand(len(env_ids), 1, 3, device=self.device)
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        self.drone.set_env_poses(pos, rot[env_ids], env_ids)
        self.drone.set_velocities(vel, env_ids)
    
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict["drone.action"]
        self.effort = self.drone.apply_action(actions)
    
    def _compute_state_and_obs(self):
        
        return TensorDict({

        }, self.batch_size)
    
    def _compute_reward_and_done(self):
        
        return TensorDict({

        }, self.batch_size)