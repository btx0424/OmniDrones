import functorch

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion
import omni.isaac.core.utils.prims as prim_utils
import torch

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec


class Hover(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.target_vis = ArticulationView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.target_vis.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        drone_state_dim = self.drone.state_spec.shape[-1]
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            UnboundedContinuousTensorSpec(drone_state_dim + 3).to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )
        self.init_pos_scale = torch.tensor([4.0, 4.0, 2.0], device=self.device)
        self.init_rpy_scale = torch.tensor([0.6, 0.6, 2.0], device=self.device) * torch.pi
        self.target_pos = torch.tensor([[0.0, 0.0, 1.5]], device=self.device)
        self.target_heading = torch.zeros(self.num_envs, 1, 3, device=self.device)

    def _design_scene(self):
        cfg = RobotCfg()
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        target_vis_prim = prim_utils.create_prim(
            prim_path="/World/envs/env_0/target",
            usd_path=self.drone.usd_path,
            translation=(0.0, 0.0, 1.5),
        )

        kit_utils.set_nested_collision_properties(
            target_vis_prim.GetPath(), 
            collision_enabled=False
        )
        kit_utils.set_nested_rigid_body_properties(
            target_vis_prim.GetPath(),
            disable_gravity=True
        )

        color = (0.0, 1.0, 0.0)
        def set_color(prim):
            if prim.GetName() == "visuals":
                prim.GetAttribute("primvars:displayColor").Set([color])
            else:
                for prim in prim.GetChildren():
                    set_color(prim)
        set_color(target_vis_prim)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.drone.spawn(translations=[(0.0, 0.0, 1.5)])
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        pos, _ = self.init_poses
        self.drone._reset_idx(env_ids)
        offset = (
            torch.rand(len(env_ids), 1, 3, device=self.device) * self.init_pos_scale
            - self.init_pos_scale / 2
        )
        rpy = (
            torch.rand(len(env_ids), 1, 3, device=self.device) * self.init_rpy_scale
            - self.init_rpy_scale / 2
        )
        
        self.drone.set_world_poses(pos[env_ids] + offset, euler_to_quaternion(rpy), env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        target_rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        # target_rpy[..., 2] = torch.rand(len(env_ids), 1, device=self.device) * 2 * torch.pi
        target_rot = euler_to_quaternion(target_rpy)
        self.target_heading[env_ids] = torch_utils.quat_axis(target_rot.squeeze(1), 0).unsqueeze(1)
        self.target_vis.set_world_poses(orientations=target_rot, env_indices=env_ids)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        # relative position and heading
        self.rpos = self.target_pos - self.root_state[..., :3]
        self.rheading = self.target_heading - self.root_state[..., 13:16]
        obs = torch.cat([
            self.rpos,
            self.root_state[..., 3:],
            self.rpos + self.rheading,
        ], dim=-1)
        return TensorDict({"drone.obs": obs}, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot, vels, heading, up = self.root_state[..., :19].split([3, 4, 6, 3, 3], dim=-1)

        # pose reward
        distance = torch.norm(torch.cat([self.rpos, self.rpos+self.rheading], dim=-1), dim=-1)

        pose_reward = 1.0 / (1.0 + torch.square(distance))
        
        # uprightness
        tiltage = torch.abs(1 - up[..., 2])
        up_reward = 1.0 / (1.0 + torch.square(tiltage))

        # effort
        effort_reward = 0.1 * torch.exp(-self.effort)

        # spin reward
        spin = torch.square(vels[..., -1])
        spin_reward = 1.0 / (1.0 + torch.square(spin))

        assert pose_reward.shape == up_reward.shape == spin_reward.shape
        reward = pose_reward + pose_reward * (up_reward + spin_reward) + effort_reward
        self._tensordict["return"] += reward.unsqueeze(-1)
        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | (pos[..., 2] < 0.1)
            | (distance > 4)
        )
        return TensorDict(
            {
                "reward": {"drone.reward": reward.unsqueeze(-1)},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )
