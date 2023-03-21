import functorch

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
import torch
from omni.isaac.core.objects import VisualSphere

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec


class Hover(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.target.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            self.drone.state_spec.to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )
        self.init_pos_scale = torch.tensor([3, 3.0, 1.0], device=self.device)

    def _design_scene(self):
        cfg = RobotCfg()
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        self.target_pos = torch.tensor([[0.0, 0.0, 1.5]], device=self.device)
        self.target = VisualSphere(
            prim_path="/World/envs/env_0/target",
            name="target",
            translation=self.target_pos,
            radius=0.05,
            color=torch.tensor([1.0, 0.0, 0.0]),
        )
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.drone.spawn(translations=[(0.0, 0.0, 1.5)])
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        pos, rot = self.init_poses
        self.drone._reset_idx(env_ids)
        offset = (
            torch.rand(len(env_ids), 1, 3, device=self.device) * self.init_pos_scale
            - self.init_pos_scale / 2
        )
        new_poses = (pos[env_ids] + offset, rot[env_ids])
        self.drone.set_world_poses(*new_poses, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict["drone.action"]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        obs = self.drone.get_state()
        obs[..., :3] = self.target_pos - obs[..., :3]
        return TensorDict({"drone.obs": obs}, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot = self.get_env_poses(self.drone.get_world_poses(False))
        vels = self.drone.get_velocities()
        # pos reward
        target_dist = torch.norm(pos - self.target_pos, dim=-1)
        pos_reward = 1.0 / (1.0 + torch.square(target_dist))
        # uprightness
        ups = functorch.vmap(torch_utils.quat_axis)(rot, axis=2)
        tiltage = torch.abs(1 - ups[..., 2])
        up_reward = 1.0 / (1.0 + torch.square(tiltage))
        # effort
        effort_reward = 0.05 * torch.exp(-0.5 * self.effort)
        # spin reward
        spin = torch.square(vels[..., -1])
        spin_reward = 1.0 / (1.0 + torch.square(spin))

        assert pos_reward.shape == up_reward.shape == spin_reward.shape
        reward = pos_reward + pos_reward * (up_reward + spin_reward)  # + effort_reward
        self._tensordict["return"] += reward.unsqueeze(-1)
        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | (pos[..., 2] < 0.1)
            | (target_dist > 3)
        )
        return TensorDict(
            {
                "reward": {"drone.reward": reward.unsqueeze(-1)},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )
