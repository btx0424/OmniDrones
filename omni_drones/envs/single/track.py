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


import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, normalize, quat_rotate_inverse
import omni.isaac.core.utils.prims as prim_utils
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import IsaacEnv
from omni_drones.envs import mdp
from omni_drones.envs.utils.trajectory import LemniscateTrajectory
from omni_drones.robots.multirotor import Multirotor
from tensordict.tensordict import TensorDict, TensorDictBase


class Track(IsaacEnv):
    r"""
    A basic control task. The goal for the agent is to track a reference 
    lemniscate trajectory in the 3D space.

    ## Observation
    
    - `rpos` (3 * `future_traj_steps`): The relative position of the drone to the 
      reference positions in the future `future_traj_steps` time steps.
    - `root_state` (16 + `num_rotors`): The basic information of the drone (except its position), 
      containing its rotation (in quaternion), velocities (linear and angular), 
      heading and up vectors, and the current throttle.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional
      vector encoding the current progress of the episode.

    ## Reward

    - `pos`: Reward for tracking the trajectory, computed from the position
      error as {math}`\exp(-a * \text{pos_error})`.
    - `up`: Reward computed from the uprightness of the drone to discourage
      large tilting.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `action_smoothness`: Reward that encourages smoother drone actions, computed based on the throttle difference of the drone.

    The total reward is computed as follows:
    ```{math}
        r = r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{heading}) + r_\text{effort} + r_\text{action_smoothness}
    ```

    ## Episode End

    The episode ends when the tracking error is larger than `reset_thres`, or
    when the drone is too close to the ground, or when the episode reaches 
    the maximum length.

    ## Config

    | Parameter               | Type  | Default       | Description |
    |-------------------------|-------|---------------|-------------|
    | `drone_model`           | str   | "hummingbird" | Specifies the model of the drone being used in the environment. |
    | `reset_thres`           | float | 0.5           | Threshold for the distance between the drone and its target, upon exceeding which the episode will be reset. |
    | `future_traj_steps`     | int   | 4             | Number of future trajectory steps the drone needs to predict. |
    | `reward_distance_scale` | float | 1.2           | Scales the reward based on the distance between the drone and its target. |
    | `time_encoding`         | bool  | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |


    """
    def __init__(self, cfg):
        self.future_traj_steps = int(cfg.task.future_traj_steps)
        assert self.future_traj_steps > 0

        super().__init__(cfg)

        self.drone: Multirotor = self.scene["drone"]

        self.init_root_state = self.drone.data.default_root_state.clone()
        self.init_joint_pos = self.drone.data.default_joint_pos.clone()
        self.init_joint_vel = self.drone.data.default_joint_vel.clone()

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        self.traj_c_dist = D.Uniform(
            torch.tensor(-0.6, device=self.device),
            torch.tensor(0.6, device=self.device)
        )
        self.traj_scale_dist = D.Uniform(
            torch.tensor([1.8, 1.8, 1.], device=self.device),
            torch.tensor([3.2, 3.2, 1.5], device=self.device)
        )
        self.traj_w_dist = D.Uniform(
            torch.tensor(0.8, device=self.device),
            torch.tensor(1.1, device=self.device)
        )
        
        self.traj_manager = LemniscateTrajectory((self.num_envs,), self.device)
        self.traj_vis = torch.zeros(self.num_envs, self.max_episode_length, 3, device=self.device)

        self.waypoints = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)
        self.target_pos = self.waypoints[:, 0]
        self.target_heading_vec = torch.zeros(self.num_envs, 3, device=self.device)

    def _design_scene(self):
        from omni.isaac.orbit.scene import InteractiveSceneCfg
        from omni.isaac.orbit.assets import AssetBaseCfg
        from omni.isaac.orbit.terrains import TerrainImporterCfg

        import omni.isaac.orbit.sim as sim_utils
        
        from omni_drones.robots.assets import HUMMINGBIRD_CFG, FIREFLY_CFG
        
        class SceneCfg(InteractiveSceneCfg):

            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
            )
            
            # lights
            light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
            )
            sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
            )
            
            drone = HUMMINGBIRD_CFG
            drone.prim_path="{ENV_REGEX_NS}/Robot_0"

        return SceneCfg(num_envs=self.cfg.num_envs, env_spacing=2.5)

    def _reset_idx(self, env_ids: torch.Tensor):
        self.traj_manager.c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_manager.rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.traj_manager.scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        traj_w = self.traj_w_dist.sample(env_ids.shape)
        self.traj_manager.w[env_ids] = torch.randn_like(traj_w).sign() * traj_w

        pos_t0 = self.traj_manager.compute(
            torch.zeros(env_ids.shape, device=self.device), 
            dt=0,
            ids=env_ids
        ).reshape(-1, 3)

        init_root_state = self.init_root_state[env_ids]
        init_root_state[..., :3] = (
            pos_t0 
            + self.scene.env_origins[env_ids] 
        )

        self.drone.write_root_state_to_sim(init_root_state, env_ids)
        self.drone.write_joint_state_to_sim(
            self.init_joint_pos[env_ids], 
            self.init_joint_vel[env_ids], 
            env_ids=env_ids
        )

        if self.sim.has_gui():
            self.traj_vis[env_ids] = (
                self.traj_manager.compute(
                    torch.zeros(env_ids.shape, device=self.device),
                    dt=self.step_dt,
                    steps=self.max_episode_length,
                    ids=env_ids
                )
                + self.scene.env_origins[env_ids].unsqueeze(1)
            )

    def update(self):
        self.waypoints[:] = self.traj_manager.compute(
            self.progress_buf * self.step_dt, 
            dt=self.step_dt * 5,
            steps=self.future_traj_steps,
        )
        self.target_heading_vec[:] = normalize(self.waypoints[:, 1] - self.waypoints[:, 0])
    
    def debug_vis(self):
        for i in range(self.num_envs):
            self.debug_draw.plot(self.traj_vis[i])
            self.debug_draw.plot(
                self.waypoints[i] + self.scene.env_origins[i].unsqueeze(0),
                size=4, 
                color=(0, 1, 0, 1)
            )
        self.debug_draw.vector(
            self.drone.data.root_pos_w,
            self.drone.data.heading_w_vec * 0.5,
            color=(1, 0, 0, 1)
        )
    
    class Waypoints(mdp.ObservationFunc):
        
        def __init__(self, env: "Track"):
            super().__init__(env)
            self.drone: Multirotor = self.env.scene["drone"]

        def compute(self) -> torch.Tensor:
            if not hasattr(self.env, "waypoints"):
                return torch.zeros(self.num_envs, 3 * self.env.future_traj_steps, device=self.device)
            pos = (
                self.drone.data.root_pos_w.unsqueeze(1)
                - self.env.scene.env_origins.unsqueeze(1)
                - self.env.waypoints
            )
            pos = quat_rotate_inverse(
                self.drone.data.root_quat_w.unsqueeze(1),
                pos
            )
            return pos.reshape(self.num_envs, -1)
    
    class PosTrackingErrorExp(mdp.RewardFunc):
        def __init__(self, env, scale: float, weight: float = 1.):
            super().__init__(env, weight)
            self.scale = scale
            self.drone: Multirotor = self.env.scene["drone"]

        def compute(self) -> torch.Tensor:
            error = (
                self.drone.data.root_pos_w
                - self.env.scene.env_origins
                - self.env.target_pos
            )
            error = torch.norm(error, dim=-1, keepdim=True)
            return torch.exp(- self.scale * error)
    
    class YawTrackingDot(mdp.RewardFunc):
        def __init__(self, env: IsaacEnv, weight: float = 1):
            super().__init__(env, weight)
            self.drone: Multirotor = self.env.scene["drone"]
        
        def compute(self) -> torch.Tensor:
            dot = (
                self.drone.data.heading_w_vec[:, :2]
                * normalize(self.drone.data.root_lin_vel_w[:, :2])
            ).sum(-1, True)
            return dot
    
    class TrackingErrorExceeds(mdp.TerminationFunc):
        def __init__(
            self,
            env: IsaacEnv,
            thres: float
        ):
            super().__init__(env)
            self.robot: Multirotor = self.env.scene["drone"]
            self.thres = thres

        def compute(self) -> torch.Tensor:
            pos_diff = (
                self.robot.data.root_pos_w
                - self.env.scene.env_origins 
                - self.env.target_pos
            )
            pos_error = pos_diff.norm(dim=-1, keepdim=True)
            terminated = (pos_error > self.thres)
            return terminated


