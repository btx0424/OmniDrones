import functorch

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
import torch
from omni.isaac.core.objects import DynamicSphere

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.utils.torch import cpos, off_diag
from omni_drones.views import RigidPrimView
from omni_drones.robots.config import RigidBodyPropertiesCfg, RobotCfg
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec

class Spread(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.targets = RigidPrimView(
            "/World/envs/env_.*/target_*",
            reset_xform_properties=False,
            shape=(-1, self.drone.n)
        )
        self.targets.initialize()

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.vels = self.drone.get_velocities()

        drone_state_dim = self.drone.state_spec.shape[0]
        observation_spec = UnboundedContinuousTensorSpec(
            drone_state_dim - 3 + self.drone.n * 6
        )
        state_spec = UnboundedContinuousTensorSpec(
            observation_spec.shape[0] * self.drone.n
        )

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.drone.n,
            observation_spec.to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
            state_spec=state_spec.to(self.device)
        )
        self.init_pos_scale = torch.tensor([4.0, 4.0, 2.0], device=self.device)
        self.init_pos_translation = torch.tensor([-2, -2, 1.5], device=self.device)
        self.target_pos = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)

    def _design_scene(self):
        cfg = RobotCfg()
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        translations = torch.zeros(self.cfg.task.num_drones, 3)
        translations[:, 1] = torch.arange(self.cfg.task.num_drones)
        self.drone.spawn(translations=translations)

        for i in range(self.drone.n):
            DynamicSphere(
                prim_path=f"/World/envs/env_0/target_{i}",
                name="target",
                translation=[0., i, 0.],
                radius=0.05,
                color=torch.tensor([1.0, 0.0, 0.0]),
            )
            kit_utils.set_rigid_body_properties(
                f"/World/envs/env_0/target_{i}",
                disable_gravity=True
            )
            kit_utils.set_collision_properties(
                f"/World/envs/env_0/target_{i}",
                collision_enabled=False
            )
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        _, rot = self.init_poses
        self.drone._reset_idx(env_ids)

        new_drone_pos = torch.rand(len(env_ids), self.drone.n, 3, device=self.device) * self.init_pos_scale + self.init_pos_translation
        new_drone_pos = new_drone_pos + self.envs_positions[env_ids].unsqueeze(1)
        self.drone.set_world_poses(new_drone_pos, rot[env_ids], env_ids)
        self.drone.set_velocities(torch.zeros_like(self.vels[env_ids]), env_ids)

        new_target_pos = torch.rand(len(env_ids), self.drone.n, 3, device=self.device) * self.init_pos_scale + self.init_pos_translation
        self.targets.set_world_poses(
            new_target_pos + self.envs_positions[env_ids].unsqueeze(1), rot[env_ids], env_ids
        )
        self.target_pos[env_ids] = new_target_pos

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict["drone.action"]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        drone_states = self.drone.get_state()
        pos = drone_states[..., :3]
        drone_relative_pos = -functorch.vmap(cpos)(pos, pos)
        target_relative_pos = -functorch.vmap(cpos)(pos, self.target_pos)

        obs = torch.cat([
            drone_relative_pos.flatten(2),
            drone_states[..., 3:],
            target_relative_pos.flatten(2)
        ], dim=-1)

        state = obs.flatten(1)

        return TensorDict({
            "drone.obs": obs,
            "drone.state": state,
        }, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot = self.get_env_poses(self.drone.get_world_poses(False))
        min_dist, idx = functorch.vmap(torch.cdist)(self.target_pos, pos).min(-1, keepdim=True)
        
        reward = torch.zeros(self.num_envs, self.drone.n, 1, device=self.device)
        reward_occ = torch.sum(1 / (1 + torch.square(min_dist)), dim=1, keepdim=True)
        tiltage = 1 - functorch.vmap(torch_utils.quat_axis)(rot, axis=2)[..., [2]]
        reward_up = 1.0 / (1.0 + torch.square(tiltage))
        spin = torch.square(self.vels[..., [-1]])
        reward_spin = 1.0 / (1.0 + torch.square(spin))

        reward[:] = reward_occ + reward_occ * (reward_up + reward_spin)

        misbehave = (pos[..., 2] < 0.25)
        done = (
            (self.progress_buf >= self.max_eposode_length).unsqueeze(-1)
            | misbehave 
        ).all(-1, keepdim=True)

        self._tensordict["return"] += reward

        return TensorDict(
            {
                "reward": {"drone.reward": reward},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )
