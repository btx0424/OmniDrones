# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.distributions as D
import einops

import omni.isaac.core.utils.prims as prim_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.terrains import (
    TerrainImporterCfg, 
    TerrainImporter, 
    TerrainGeneratorCfg,
    HfDiscreteObstaclesTerrainCfg,
)
from omni.isaac.lab.utils.assets import NVIDIA_NUCLEUS_DIR
from omni.isaac.core.utils.viewports import set_camera_view


class Forest(IsaacEnv):
    r"""
    This is a single-agent task where the agent is required to navigate a randomly 
    generated cluttered environment. The agent needs to fly at a commanded speed 
    along the positive direction while avoiding collisions with obstacles.

    The agent utilizes Lidar sensors to perceive its surroundings. The Lidar has 
    a horizontal field of view (FOV) of 360 degrees and a the vertical view can be 
    specified.

    ## Observation

    The observation is given by a `CompositeSpec` containing the following values:

    - `"state"` (16 + `num_rotors`): The basic information of the drone 
      (except its position), containing its rotation (in quaternion), velocities 
      (linear and angular), heading and up vectors, and the current throttle.
    - `"lidar"` (1, w, h) : The lidar scan of the drone. The size is decided by the
      field of view and resolution.

    ## Reward

    - `vel`: Reward computed from the position error to the target position.
    - `up`: Reward computed from the uprightness of the drone to discourage large tilting.
    - `survive`: Reward of a constant value to encourage collision avoidance.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `action_smoothness`: Reward that encourages smoother drone actions, 
      computed based on the throttle difference of the drone.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{vel} + r_\text{up} + r_\text{survive} + r_\text{effort} + r_\text{action_smoothness}
    ```
        
    ## Episode End

    The episode ends when the drone mishebaves, e.g., when the drone collides 
    with the ground or obstacles, or when the drone flies out of the boundar:

    ```{math}
        d_\text{ground} < 0.2 \text{ or } d_\text{ground} > 4.0 \text{ or } v_\text{drone} > 2.5
    ```
    
    or when the episode reaches the maximum length.


    ## Config

    | Parameter               | Type  | Default   | Description |
    |-------------------------|-------|-----------|-------------|
    | `drone_model`           | str   | "firefly" | Specifies the model of the drone being used in the environment. |
    | `lidar_range`           | float | 4.0       | Specifies the maximum range of the lidar. |
    | `lidar_vfov`            | float | [-10, 20] | Specifies the vertical field of view of the lidar. |
    | `time_encoding`         | bool  | True      | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |


    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()

        super().__init__(cfg, headless)
        
        self.lidar_vfov = (
            max(-89., cfg.task.lidar_vfov[0]), 
            min(89., cfg.task.lidar_vfov[1])
        )
        self.lidar_range = cfg.task.lidar_range
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=False,
            pattern_cfg=patterns.BpearlPatternCfg(
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, 4)
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.lidar: RayCaster = ray_caster_cfg.class_type(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.lidar_resolution = (36, 4)

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])
        
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )

        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = 24.
            self.target_pos[:, 0, 2] = 2.

        self.alpha = 0.8

    def _design_scene(self):
        import omni_drones.utils.kit as kit_utils
        from pxr import PhysxSchema, UsdPhysics
        from omni_drones.utils.poisson_disk import poisson_disk_sampling

        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.)])[0]

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        rot = euler_to_quaternion(torch.tensor([0., 0.1, 0.1]))
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos, rot)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(8.0, 8.0),
                border_width=20.0,
                num_rows=5,
                num_cols=5,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        size=(8.0, 8.0),
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=40,
                        obstacle_height_mode="choice",
                        obstacle_width_range=(0.4, 0.8),
                        obstacle_height_range=(3.0, 4.0),
                        platform_width=1.5,
                    )
                },
            ),
            max_init_terrain_level=5,
            collision_group=-1,
            # visual_material=sim_utils.MdlFileCfg(
            #     mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            #     project_uvw=True,
            # ),
            debug_vis=False,
        )
        terrain: TerrainImporter = terrain_cfg.class_type(terrain_cfg)

        return ["/World/ground"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device),
                    "lidar": UnboundedContinuousTensorSpec((1, 36, 4), device=self.device),
                }),
                "intrinsics": self.drone.intrinsics_spec.to(self.device)
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec,
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
            "safety": UnboundedContinuousTensorSpec(1)
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "prev_action": self.drone.action_spec.to(self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        
        pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
        pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
        pos[:, 0, 1] = -24.
        pos[:, 0, 2] = 2.

        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos, rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions.unsqueeze(1))

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.lidar.update(self.dt) 
        
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state(env_frame=False)
        self.info["drone_state"][:] = self.root_state[..., :13]
        # relative position and heading
        self.rpos = self.target_pos - self.root_state[..., :3]
        
        self.lidar_scan = self.lidar_range - (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        )

        distance = self.rpos.norm(dim=-1, keepdim=True)
        rpos_clipped = self.rpos / distance.clamp(1e-6)
        obs = {
            "state": torch.cat([rpos_clipped, self.root_state[..., 3:]], dim=-1).squeeze(1),
            "lidar": self.lidar_scan
        }

        if self._should_render(0):
            self.debug_draw.clear()
            x = self.lidar.data.pos_w[0]
            set_camera_view(
                eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
                target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)                        
            )
            v = (self.lidar.data.ray_hits_w[0] - x).reshape(*self.lidar_resolution, 3)
            self.debug_draw.vector(x.expand_as(v[:, 0]), v[:, 0])
            self.debug_draw.vector(x.expand_as(v[:, -1]), v[:, -1])

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        # pose reward
        distance = self.rpos.norm(dim=-1, keepdim=True)
        vel_direction = self.rpos / distance.clamp_min(1e-6)

        reward_safety = torch.log(self.lidar_range-self.lidar_scan).mean(dim=(2, 3))
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1).clip(max=2.0)
        
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)

        # effort
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)

        reward = reward_vel + reward_up + 1. + reward_safety * 0.2
        

        terminated = (
            (self.drone.pos[..., 2] < 0.2) 
            | (self.drone.pos[..., 2] > 4.)
            | (self.drone.vel_w[..., :3].norm(dim=-1) > 2.5)
            | (einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") >  (self.lidar_range - 0.3))
        )
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["safety"].add_(reward_safety)
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated
            },
            self.batch_size,
        )
