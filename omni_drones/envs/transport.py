import torch
import functorch
from torchrl.data import UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict, TensorDictBase

import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import objects

from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils


class Transport(IsaacEnv):
    
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize("/World/envs/env_.*/Unit_.*/Firefly")
        self.init_poses = self.drone.get_env_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.agent_spec["drone"] = AgentSpec(
            "drone", 4, 
            self.drone.state_spec.to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )

    def _design_scene(self):
        cfg = RobotCfg()
        self.drone: MultirotorBase = MultirotorBase.REGISTRY["Firefly"](cfg=cfg)

        scene_utils.design_scene()
        payload = objects.DynamicCuboid(
            prim_path="/World/envs/env_0/payload",
            position=(0., 0., 0.4), 
            scale=(0.9, 0.9, 0.45),
            mass=2.
        )
        payload.set_collision_enabled(False)

        translations = [
            [-.5, .5, 1.5],
            [ .5, .5, 1.5],
            [ .5, -.5, 1.5],
            [-.5, -.5, 1.5]
        ]

        for i in range(4):
            unit = prim_utils.create_prim(
                f"/World/envs/env_0/Unit_{i}",
                translation=translations[i]
            )
            prims = self.drone.spawn(
                translations=[(0., 0., 0.)],
                prim_paths=[f"/World/envs/env_0/Unit_{i}/Firefly"]
            )
            scene_utils.create_rope(
                f"/World/envs/env_0/Unit_{i}/rope_{i}", 
                from_prim=payload.prim,
                to_prim=f"/World/envs/env_0/Unit_{i}/Firefly/base_link",
            )
        
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        return
        pos, rot = self.init_poses
        self.drone._reset_idx(env_ids)
        
        self.drone.set_env_poses(pos[env_ids], rot[env_ids], env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict["drone.action"]
        self.effort = self.drone.apply_action(actions)
    
    def _compute_state_and_obs(self):
        obs = self.drone.get_state()

        return TensorDict({
            "drone.obs": obs
        }, self.batch_size)
    
    def _compute_reward_and_done(self):
        pos, rot = self.drone.get_env_poses(False)
        reward = torch.zeros(self.num_envs, self.drone._count, 1, device=self.device)
        self._tensordict["drone.return"] += reward
        done  = (
            (self.progress_buf >= self.max_eposode_length).unsqueeze(-1)
        )
        return TensorDict({
            "reward": {
                "drone.reward": reward
            },
            "done": done
        }, self.batch_size)

