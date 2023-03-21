import functorch

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
import omni.isaac.core.utils.prims as prim_utils
import torch
from pxr import PhysxSchema, UsdPhysics

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from omni_drones.utils.poisson_disk import poisson_disk_sampling
from omni_drones.utils.scene import design_scene
from omni_drones.views import XFormPrimView


class Forest(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        
        self.trees = XFormPrimView(
            "/World/envs/env_.*/tree_*",
            reset_xform_properties=False
        )
        self.trees.initialize()
        self.trees_pos = self.trees.get_world_poses()[0].reshape(self.num_envs, -1, 3) - self.envs_positions.unsqueeze(1)

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            self.drone.state_spec.to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )
        self.target_pos = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)

        info_spec = CompositeSpec(
            trees_pos=UnboundedContinuousTensorSpec((self.trees_pos.shape[1], 2)),
            target_pos=UnboundedContinuousTensorSpec((self.drone.n, 3)),
            time=UnboundedContinuousTensorSpec(1),
        ).expand(self.num_envs).to(self.device)

        self.observation_spec["info"] = info_spec

    def _design_scene(self):
        cfg = RobotCfg()
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        self.forest_size = (length, width, r) = 16, 12, 1.8
        trees = poisson_disk_sampling(length, width, r) + torch.tensor([2., -width/2])

        for i, pos in enumerate(trees):
            tree_prim = prim_utils.create_prim(
                f"/World/envs/env_0/tree_{i}",
                prim_type="Capsule",
                attributes={"radius": 0.1, "height": 4.0},
                translation=(*pos, 2),
            )
            UsdPhysics.CollisionAPI.Apply(tree_prim)
            PhysxSchema.PhysxCollisionAPI.Apply(tree_prim)
        
        design_scene()

        self.drone.spawn(translations=[(0.0, 0.0, 1.5)])
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        _, rot = self.init_poses
        self.drone._reset_idx(env_ids)
        
        pos = torch.zeros((len(env_ids), self.drone.n, 3), device=self.device)
        pos[..., 1].uniform_(-self.forest_size[1]/4, self.forest_size[1]/4)
        pos[..., 2] = 1.5
        new_poses = (pos + self.envs_positions[env_ids].unsqueeze(1), rot[env_ids])
        self.drone.set_world_poses(*new_poses, env_ids)
        self.drone.set_velocities(torch.zeros_like(self.init_vels[env_ids]), env_ids)

        target_pos = torch.zeros_like(pos)
        target_pos[..., 0] = self.forest_size[0] + 4
        target_pos[..., 1].uniform_(-self.forest_size[1]/4, self.forest_size[1]/4)
        target_pos[..., 2] = 1.5
        self.target_pos[env_ids] = target_pos

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict["drone.action"]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        obs = self.drone.get_state()
        
        tensordict = TensorDict({
            "drone.obs": obs,
            "info":{
                "target_pos": self.target_pos.clone(),
                "trees_pos": self.trees_pos[..., :2].clone(),
                "time": (self.progress_buf * self.dt).unsqueeze(-1),
            }
        }, self.batch_size)
        return tensordict

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
        )
        return TensorDict(
            {
                "reward": {"drone.reward": reward.unsqueeze(-1)},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )
