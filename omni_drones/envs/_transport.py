import functorch
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.core.utils.torch as torch_utils
import torch
from omni.isaac.core import objects
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase


class Transport(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize("/World/envs/env_.*/Firefly_*")
        self.init_poses = self.drone.get_env_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            4,
            self.drone.state_spec.to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )

    def _design_scene(self):
        cfg = RobotCfg()
        self.drone: MultirotorBase = MultirotorBase.REGISTRY["Firefly"](cfg=cfg)

        scene_utils.design_scene()
        prim_utils.create_prim(
            "/World/envs/env_0/Payload",
            position=(0.0, 0.0, 0.4),
        )
        payload = objects.DynamicCuboid(
            prim_path="/World/envs/env_0/Payload/Cube", scale=(0.9, 0.9, 0.45), mass=2.0
        )
        # payload.set_collision_enabled(False)

        translations = torch.tensor(
            [[-0.5, 0.5, 1.5], [0.5, 0.5, 1.5], [0.5, -0.5, 1.5], [-0.5, -0.5, 1.5]]
        )

        for i in range(4):
            self.drone.spawn(
                translations=translations[i],
                prim_paths=[f"/World/envs/env_0/Firefly_{i}"],
            )

            scene_utils.create_bar(
                f"/World/envs/env_0/Firefly_{i}/bar",
                length=1.0,
                translation=(0, 0, -0.5),
                # from_prim=payload.prim,
                to_prim=f"/World/envs/env_0/Firefly_{i}/base_link",
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

        return TensorDict({"drone.obs": obs}, self.batch_size)

    def _compute_reward_and_done(self):
        # pos, rot = self.drone.get_env_poses(False)
        reward = torch.zeros(self.num_envs, self.drone.n, 1, device=self.device)
        self._tensordict["return"] += reward
        done = (self.progress_buf >= self.max_eposode_length).unsqueeze(-1)
        return TensorDict(
            {
                "reward": {"drone.reward": reward},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )
