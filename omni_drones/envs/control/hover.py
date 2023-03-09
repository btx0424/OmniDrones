import torch
import functorch
from torchrl.data import UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict, TensorDictBase

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.objects import VisualSphere

from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.config import RobotCfg, RigidBodyPropertiesCfg
from omni_drones.robots.drone import MultirotorBase
import omni_drones.utils.kit as kit_utils


class Hover(IsaacEnv):
    
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.target.initialize()
        self.init_poses = self.drone.get_env_poses(clone=True)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1, 
            self.drone.state_spec.to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )
        self.vels = self.drone.get_velocities()
        self.init_pos_scale = torch.tensor([2., 2., 0.6], device=self.device) 
        self.init_pos_offset = torch.tensor([-1., -1., 0.4], device=self.device)

    def _design_scene(self):
        cfg = RobotCfg()
        self.drone = MultirotorBase.REGISTRY["Firefly"](cfg=cfg)

        self.target_pos = torch.tensor([[0., 0., 1.5]], device=self.device)
        self.target = VisualSphere(
            prim_path="/World/envs/env_0/target",
            name="target",
            translation=self.target_pos,
            radius=0.05, 
            color=torch.tensor([1., 0., 0.])
        )
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.drone.spawn(translations=[(0., 0., 1.)])
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        _, rot = self.init_poses
        self.drone._reset_idx(env_ids)
        pos = torch.rand(len(env_ids), 1, 3, device=self.device) * self.init_pos_scale + self.init_pos_offset
        self.drone.set_env_poses(pos, rot[env_ids], env_ids)
        self.drone.set_velocities(torch.zeros_like(self.vels[env_ids]), env_ids)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict["drone.action"]
        self.effort = self.drone.apply_action(actions)
    
    def _compute_state_and_obs(self):
        obs = self.drone.get_state()
        obs[..., :3] = self.target_pos - obs[..., :3]
        return TensorDict({
            "drone.obs": obs
        }, self.batch_size)
    
    def _compute_reward_and_done(self):
        pos, rot = self.drone.get_env_poses(False)
        # pos reward        
        target_dist = torch.norm(pos-self.target_pos, dim=-1)
        pos_reward = 1.0 / (1.0 + torch.square(target_dist))
        # uprightness
        ups = functorch.vmap(torch_utils.quat_axis)(rot, axis=2)
        tiltage = torch.abs(1 - ups[..., 2])
        up_reward = 1.0 / (1.0 + torch.square(tiltage))
        # effort
        effort_reward = 0.05 * torch.exp(-0.5 * self.effort)
        # spin reward
        spin = torch.square(self.vels[..., -1])
        spin_reward = 1.0 / (1.0 + torch.square(spin))

        assert pos_reward.shape == up_reward.shape == spin_reward.shape
        reward = pos_reward + pos_reward * (up_reward + spin_reward) # + effort_reward
        self._tensordict["return"] += reward.unsqueeze(-1)
        done  = (
            (self.progress_buf >= self.max_eposode_length).unsqueeze(-1)
            | (pos[..., 2] < 0.1)
            | (target_dist > 3)
        )
        return TensorDict({
            "reward": {
                "drone.reward": reward.unsqueeze(-1)
            },
            "return": self._tensordict["return"],
            "done": done
        }, self.batch_size)

