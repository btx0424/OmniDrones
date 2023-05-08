from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import torch

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.utils.torch import cpos, off_diag, others, make_cells
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

def sample_from_grid(cells: torch.Tensor, n):
    idx = torch.randperm(cells.shape[0], device=cells.device)[:n]
    return cells[idx]

class Formation(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)

        drone_state_dim = self.drone.state_spec.shape[0]
        observation_spec = CompositeSpec({
            "state_self": UnboundedContinuousTensorSpec((1, drone_state_dim)).to(self.device),
            "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, drone_state_dim+1)).to(self.device),
        })

        state_spec = CompositeSpec(
            {
                "drones": self.drone.state_spec.expand(
                    self.drone.n, *self.drone.state_spec.shape
                ).to(self.device),
            }
        )

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            5,
            observation_spec,
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
            state_spec,
        )

        # initial state distribution
        self.target_pos = torch.tensor(self.num_envs, self.drone.n, 3, device=self.device)
        
        self.alpha = 0.7
        # additional infos & buffers
        stats_spec = CompositeSpec({
            "pos_error": UnboundedContinuousTensorSpec(1),
            "effort": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1)
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = state_spec.zero()

    def _design_scene(self) -> Optional[List[str]]:
        cfg = RobotCfg()
        drone_cls = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        self.drone: MultirotorBase = drone_cls(cfg=cfg)

        scene_utils.design_scene()

        n = cfg.task.size
        x = torch.linspace(-n/2, n/2, n)
        y = torch.linspace(-n/2, n/2, n)
        zz = torch.ones(n, n) * 2
        translations = torch.stack([*torch.meshgrid(x, y), zz], dim=-1)
        self.drone.spawn(translations=translations)

        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)


    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.drone_pos = self.drone_state[..., :3]

        target_drone_rpos = self.target_pos - self.drone_pos
        drone_rpos = vmap(cpos)(self.drone_pos, self.drone_pos)
        self.pdist = vmap(off_diag)(torch.norm(drone_rpos, dim=-1, keepdim=True))
        drone_rpos = vmap(off_diag)(drone_rpos)

        state_others = torch.cat([
            drone_rpos,
            self.pdist,
            vmap(others)(self.drone_state[..., 3:])
        ], dim=-1)

        obs = TensorDict(
            {
                "state_self": states.unsqueeze(2),
                "state_others": state_others,
                "target": states[..., :3].unsqueeze(2),
            },
            [self.num_envs, self.drone.n],
        )

        state = TensorDict({"drones": states}, self.batch_size)

        return TensorDict({
            "drone.obs": obs, 
            "drone.state": state,
        }, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot = self.get_env_poses(self.drone.get_world_poses())


        self._tensordict["return"] += reward

        terminated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        crash = (pos[..., 2] < 0.2).any(-1, keepdim=True)

        done = terminated | crash

        return TensorDict(
            {
                "reward": {"drone.reward": reward},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )

