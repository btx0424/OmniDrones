from typing import List, Optional

from omni.isaac.core.utils.viewports import set_camera_view
import omni.isaac.orbit.utils.kit as kit_utils

from omni.isaac.orbit.robots.legged_robot import LeggedRobot
from omni.isaac.orbit.robots.config.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG
from omni.isaac.orbit.robots.config.unitree import UNITREE_A1_CFG
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit_envs.locomotion.velocity.velocity_cfg import ObservationsCfg, RewardsCfg
from omni_drones.envs import IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import quat_axis, normalize, quat_rotate_inverse
import torch
import torch.distributions as D
import numpy as np

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, BinaryDiscreteTensorSpec

from omni.isaac.debug_draw import _debug_draw

class VelocityEnv(IsaacEnv):
    def __init__(self, cfg, headless):
        self.time_encoding = cfg.task.time_encoding

        super().__init__(cfg, headless)
        # -- history
        self.actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        self.init_vel_dist = D.Uniform(
            torch.tensor([-.5, -.5, 0.], device=self.device),
            torch.tensor([.5, .5, 0.], device=self.device)
        )
        # -- command: x vel, y vel, yaw vel, heading
        self.commands_dist = D.Uniform(
            torch.tensor([-1.3, -1.3], device=self.device), 
            torch.tensor([1.3, 1.3], device=self.device),
        )
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_interval = 400
        self.push_interval = 600
        self.push_acc_dist = D.Uniform(
            torch.tensor([-3, -3, -0.25], device=self.device) / (self.dt * self.substeps), 
            torch.tensor([3., 3., 0.25], device=self.device) / (self.dt * self.substeps),
        )
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.filtered_linvel_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.robot.base = RigidPrimView(
            "/World/envs/env_*/Robot/base",
            reset_xform_properties=False,
        )
        self.robot.base.initialize()
        self.base_mass = self.robot.base.get_masses(clone=True)
        self.base_mass_dist = D.Uniform(
            torch.tensor([0.8], device=self.device),
            torch.tensor([1.4], device=self.device)
        )
        
        self.base_target_height = 0.3

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        # self.robot = LeggedRobot(ANYMAL_C_CFG)
        self.robot = LeggedRobot(UNITREE_A1_CFG)
        self.robot.spawn("/World/envs/env_0/Robot")
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            improve_patch_friction=True,
            combine_mode="max",
        )
        return ["/World/defaultGroundPlane"]
    
    def _set_specs(self):
        self.robot.initialize("/World/envs/env_*/Robot")
        observation_dim = 52
        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "intrinsics": CompositeSpec({
                    "base_mass": UnboundedContinuousTensorSpec((1, 1), device=self.device)
                })
            },
            "truncated": BinaryDiscreteTensorSpec(1, dtype=bool, device=self.device)
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((1, self.robot.num_actions)),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "lin_vel_error": UnboundedContinuousTensorSpec(1),
            "dof_torques": UnboundedContinuousTensorSpec(1),
            "dof_acc": UnboundedContinuousTensorSpec(1),
            "action_rate": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()
        self.intrinsics = self.observation_spec[("agents", "intrinsics")].zero()


    def _reset_idx(self, env_ids: torch.Tensor):
        # -- dof state (handled by the robot)
        dof_pos, dof_vel = self.robot.get_random_dof_state(env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        # -- root state (custom)
        root_state = self.robot.get_default_root_state(env_ids)
        root_state[:, :3] += self.envs_positions[env_ids]
        root_state[:, 7:10] = self.init_vel_dist.sample(env_ids.shape)
        root_state[:, 10:13] = 0.
        # set into robot
        self.robot.set_root_state(root_state, env_ids=env_ids)
        self.robot.reset_buffers(env_ids)
        self.filtered_linvel_w[env_ids] = 0.

        # -- reset history
        self.previous_actions[env_ids] = 0.
        self.stats[env_ids] = 0.

        # sample commands
        lin_vel_commands = self.commands_dist.sample(env_ids.shape)
        self.commands[env_ids, :2] = lin_vel_commands

        # sample base mass
        base_mass = self.base_mass_dist.sample(env_ids.shape)
        self.intrinsics["base_mass"][env_ids] = base_mass.unsqueeze(-1)
        self.robot.base.set_masses(base_mass * self.base_mass[env_ids], env_indices=env_ids)
    
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.actions[:] = tensordict[("agents", "action")].squeeze(1)
        # push = ((self.progress_buf % self.push_interval) == 0).nonzero().squeeze(1)
        # if len(push) > 0:
        #     push_force = self.base_mass[push] * self.push_acc_dist.sample(push.shape)
        for substep in range(self.substeps):
            self.robot.apply_action(self.actions.clip(-100., 100.))
            # if len(push) > 0:
            #     self.robot.base.apply_forces(push_force, indices=push)
            self.sim.step(self._should_render(substep))
        self._post_sim_step(tensordict)
        self.progress_buf += 1
        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(self._compute_state_and_obs())
        tensordict.update(self._compute_reward_and_done())
        return tensordict
    
    def _compute_state_and_obs(self):
        self.robot.update_buffers(dt=self.dt * self.substeps)
        self.robot.data.heading = quat_axis(self.robot.data.root_quat_w, 0)

        obs = [
            self.robot.data.root_pos_w[:, [2]],
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_lin_vel_w,
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            quat_rotate_inverse(self.robot.data.root_quat_w, self.commands),
            self.robot.data.dof_pos - self.robot.data.actuator_pos_offset,
            self.robot.data.dof_vel - self.robot.data.actuator_vel_offset,
            self.actions,
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim))
        obs = torch.cat(obs, dim=-1)

        if self._should_render(0):
            self.draw.clear_lines()
            colors = [(1.0, 1.0, 1.0, 1.0), (1.0, 0.5, 0.5, 1.0)]
            sizes = [2, 2]
            robot_pos = self.robot.data.root_pos_w[self.central_env_idx].cpu()
            linvel_start = robot_pos + torch.tensor([0., 0., 0.5])
            linvel_end = linvel_start + self.robot.data.root_lin_vel_w[self.central_env_idx].cpu()
            target_linvel_end = linvel_start + self.commands[self.central_env_idx].cpu()
            
            point_list_0 = [ linvel_start.tolist(), linvel_start.tolist()]
            point_list_1 = [target_linvel_end.tolist(), linvel_end.tolist()]
            
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)
            set_camera_view(
                eye=robot_pos.numpy() + np.asarray(self.cfg.viewer.eye),
                target=robot_pos.numpy() + np.asarray(self.cfg.viewer.lookat)                        
            )
    

        return TensorDict({
            "agents": {
                "observation": obs.unsqueeze(1),
                "intrinsics": self.intrinsics,
            },
            "stats": self.stats.clone(),
        }, self.num_envs)

    def _compute_reward_and_done(self):
        # -- compute reward
        lin_vel_error = square_norm(self.commands[:, :2] - self.robot.data.root_lin_vel_w[:, :2])
        # ang_vel_error = torch.square(self.commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        base_height_error = (self.robot.data.root_pos_w[:, 2] - self.base_target_height).abs()
        heading_projection = (normalize(self.robot.data.heading[:, :2]) * self.commands[:, :2]).sum(-1, keepdim=True)

        lin_vel_xy_exp = torch.exp(-lin_vel_error / 0.25)
        # ang_vel_z_exp = torch.exp(-ang_vel_error / 0.25)
        lin_vel_z_l2 = torch.square(self.robot.data.root_lin_vel_w[:, 2])
        ang_vel_xy_l2 = square_norm(self.robot.data.root_ang_vel_w[:, :2])
        flat_orientation_l2 = square_norm(self.robot.data.projected_gravity_b[:, :2])
        dof_torques_l2 = square_norm(self.robot.data.applied_torques)
        dof_acc_l2 = square_norm(self.robot.data.dof_acc)
        action_rate_l2 = square_norm(self.previous_actions - self.actions)
        self.previous_actions[:] = self.actions
        
        reward = (
            2.0 * lin_vel_xy_exp
            + 0.5 * heading_projection
            # + 0.5 * ang_vel_z_exp.unsqueeze(1)
            + 0.25 / (1 + base_height_error).unsqueeze(1)
            - 2.0 * lin_vel_z_l2.unsqueeze(1)
            - 0.05 * ang_vel_xy_l2
            - 2.0 * flat_orientation_l2
            - 0.000025 * dof_torques_l2
            - 2.5e-7 * dof_acc_l2
            - 0.01 * action_rate_l2
        ).clip(min=0.)


        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        done = (
            self.robot.data.root_pos_w[:, 2] <= self.base_target_height * 0.5
        ).unsqueeze(1)
        
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["lin_vel_error"].add_(lin_vel_error)
        self.stats["dof_torques"].add_(dof_torques_l2)
        self.stats["dof_acc"].add_(dof_acc_l2)
        self.stats["action_rate"].add_(action_rate_l2)

        # resample commands
        change_commands = ((self.progress_buf % self.command_interval) == 0).nonzero().squeeze(1)
        self.commands[change_commands, :2] = self.commands_dist.sample(change_commands.shape)

        return TensorDict({
            "agents": {
                "reward": reward.reshape(-1, 1, 1)
            },
            "done": done,
            "truncated": truncated,
        }, self.num_envs)

def square_norm(x: torch.Tensor):
    return x.square().sum(dim=-1, keepdim=True)

