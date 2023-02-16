import torch
from torchrl.data import UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict, TensorDictBase
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.drone.crazyflie import Crazyflie
from omni_drones.robots.drone.quadcopter import Quadcopter

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.core.objects import VisualSphere

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

    def _design_scene(self):
        self.target_pos = torch.tensor([0., 0., 1.0], device=self.device)
        self.drone = Crazyflie()
        # self.drone = Quadcopter()
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
        self.drone.spawn(translation=(0., 0., 1.0))
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        pos, rot = self.init_poses
        self.drone._reset_idx(env_ids)
        self.drone.set_env_poses(pos[env_ids], rot[env_ids], env_ids)
        self.drone.set_velocities(torch.zeros_like(self.vels[env_ids]), env_ids)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        self.effort = self.drone.apply_action(tensordict["drone.action"])
    
    def _compute_state_and_obs(self):
        self.pos, self.rot = self.drone.get_env_poses(True)
        self.vels = self.drone.get_velocities()
        pos = self.target_pos - self.pos
        dof_pos = self.drone.get_joint_positions()
        obs = torch.cat([pos, self.rot, self.vels, dof_pos], dim=-1)
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

        reward = pos_reward + pos_reward * (spin_reward) # + effort_reward
        self._tensordict["drone.return"] += reward.unsqueeze(-1)
        done  = (
            (self.progress_buf >= self.max_eposode_length).unsqueeze(-1)
            | (self.pos[..., 2] < 0.2)
            | (self.pos[..., 2] > 3.0)
        )
        return TensorDict({
            "reward": {
                "drone.reward": reward.unsqueeze(-1)
            },
            "done": done
        }, self.batch_size)

