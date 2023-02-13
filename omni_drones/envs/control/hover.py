import torch
from torchrl.data import UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict, TensorDictBase
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.drone.crazyflie import Crazyflie

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.core.objects import VisualSphere

class Hover(IsaacEnv):
    
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.target.initialize()
        self.init_poses = self.drone.get_env_poses()
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1, 
            self.drone.state_spec.to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )

    def _design_scene(self):
        self.target_pos = torch.tensor([0., 0., 1.0], device=self.device)
        self.drone = Crazyflie()
        self.target = VisualSphere(
            prim_path="/World/envs/env_0/target",
            name="target",
            translation=self.target_pos,
            radius=0.1, color=torch.tensor([1., 0., 0.])
        )
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.drone.spawn(translation=(0., 0., 0.5))
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, mask: torch.Tensor):
        pos, rot = self.init_poses
        self.drone.set_env_poses(pos, rot, mask.nonzero().squeeze(-1))

    def _pre_sim_step(self, tensordict: TensorDictBase):
        self.effort = self.drone.apply_action(tensordict["drone.action"])
    
    def _compute_state_and_obs(self):
        self.pos, self.rot = self.drone.get_env_poses()
        self.vels = self.drone.get_velocities()
        pos = self.target_pos - self.pos
        obs = torch.cat([pos, self.rot, self.vels], dim=-1)
        return TensorDict({
            "drone.obs": obs
        }, self.batch_size)
    
    def _compute_reward_and_done(self):
        # pos reward        
        target_dist = torch.norm(self.pos - self.target_pos)
        pos_reward = 1.0 / (1.0 + target_dist)
        # uprightness

        # effort
        effort_reward = 0.05 * torch.exp(-0.5 * self.effort)
        # spin reward
        spin = torch.square(self.vels[..., 3:]).sum(-1)
        spin_reward = 0.01 * torch.exp(-1.0 * spin)

        reward = pos_reward + pos_reward * (spin_reward) + effort_reward
        return TensorDict({
            "reward": {
                "drone.reward": reward.unsqueeze(-1)
            },
            "done": (self.progress_buf >= self.max_eposode_length).clone()
        }, self.batch_size)

