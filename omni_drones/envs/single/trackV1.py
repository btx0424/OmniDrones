from functorch import vmap
import math
import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, normalize
import omni.isaac.core.utils.prims as prim_utils
import torch
import torch.distributions as D

import numpy as np

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from omni.isaac.debug_draw import _debug_draw

from ..utils import lemniscate, scale_time


class TimeEncoding:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, t: torch.Tensor):
        ...


class Fraction(TimeEncoding):
    def __init__(self, max_episode_len, dim: int=4):
        super().__init__(dim)
        self.max_episode_len = max_episode_len
    
    def encode(self, t: torch.Tensor):
        _t = (t / self.max_episode_len).unsqueeze(-1)
        return _t.expand(*t.shape, self.dim)


class Fourier(TimeEncoding):
    def __init__(self, dim: int, seed=0) -> None:
        super().__init__(dim)
    
    def encode(self, t: torch.Tensor):
        ...


def attach_payload(parent_path):
    from omni.isaac.core import objects
    import omni.physx.scripts.utils as script_utils
    from pxr import UsdPhysics

    payload_prim = objects.DynamicCuboid(
        prim_path=parent_path + "/payload",
        scale=torch.tensor([0.1, 0.1, .15]),
        mass=0.0001
    ).prim

    parent_prim = prim_utils.get_prim_at_path(parent_path + "/base_link")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "Prismatic", payload_prim, parent_prim)
    UsdPhysics.DriveAPI.Apply(joint, "linear")
    joint.GetAttribute("physics:lowerLimit").Set(-0.15)
    joint.GetAttribute("physics:upperLimit").Set(0.15)
    joint.GetAttribute("physics:axis").Set("Z")
    joint.GetAttribute("drive:linear:physics:damping").Set(10.)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(10000.)

    
class TrackV1(IsaacEnv):
    """
    The UAV need to track a reference trajectory while keeping its heading direction aligned with the flight direction. 
    
    """
    def __init__(self, cfg, headless):
        self.reset_thres = cfg.task.reset_thres
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.future_traj_steps = max(int(cfg.task.future_traj_steps), 2)
        self.wind = cfg.task.wind
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()
        self.device_name='cuda:0'

        super().__init__(cfg, headless)
        
        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])
        if "payload" in self.randomization:
            payload_cfg = self.randomization["payload"]
            self.payload_z_dist = D.Uniform(
                torch.tensor([payload_cfg["z"][0]], device=self.device),
                torch.tensor([payload_cfg["z"][1]], device=self.device)
            )
            self.payload_mass_dist = D.Uniform(
                torch.tensor([payload_cfg["mass"][0]], device=self.device),
                torch.tensor([payload_cfg["mass"][1]], device=self.device)
            )
            self.low=torch.tensor([payload_cfg["mass"][0]], device=self.device)
            self.high=torch.tensor([payload_cfg["mass"][1]], device=self.device)
            self.payload = RigidPrimView(
                f"/World/envs/env_*/{self.drone.name}_*/payload",
                reset_xform_properties=False,
                shape=(-1, self.drone.n)
            )
            self.payload.initialize()
            
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        self.traj_c_dist = D.Uniform(
            torch.tensor(-0.8, device=self.device),
            torch.tensor(0.8, device=self.device)
        )
        self.traj_scale_dist = D.Uniform(
            torch.tensor([2.4, 2.4, 1.2], device=self.device),
            torch.tensor([4., 4., 1.6], device=self.device)
        )
        self.traj_w_dist = D.Uniform(
            torch.tensor(0.8, device=self.device),
            torch.tensor(1.6, device=self.device)
        )
        self.origin = torch.tensor([0., 0., 2.5], device=self.device)

        self.traj_t0 = torch.pi / 2
        self.traj_c = torch.zeros(self.num_envs, device=self.device)
        self.traj_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.traj_w = torch.ones(self.num_envs, device=self.device)

        self.ref_pos = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)
        self.ref_heading = torch.zeros(self.num_envs, 2, device=self.device)

        self.alpha = 0.8
        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.5)])[0]
        if self.has_payload:
            attach_payload(drone_prim.GetPath().pathString)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_obs_dim = (
            self.drone.state_spec.shape[-1] 
            + 3 * (self.future_traj_steps-1) 
            + 2 # reference xy heading
            + self.drone.action_spec.shape[-1] # last action
        )
        if self.time_encoding:
            self.time_encoding = Fraction(self.max_episode_length)
            drone_obs_dim += self.time_encoding.dim
        
        # TODO@btx0424: observe history through a Transform
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, drone_obs_dim), device=self.device),
                "observation_h": UnboundedContinuousTensorSpec((1, 27, 50), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((1, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_key={
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "tracking_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
            "dist_one":UnboundedContinuousTensorSpec(1),
            "dist_two":UnboundedContinuousTensorSpec(1),
            "dist_three":UnboundedContinuousTensorSpec(1),
            "dist_four":UnboundedContinuousTensorSpec(1),
            "dist_five":UnboundedContinuousTensorSpec(1),
        }
        '''for i in range(self.num_envs):
            mse_key=f"mse_{i}"
            stats_key[mse_key]=UnboundedContinuousTensorSpec(1)'''
        stats_spec = CompositeSpec(stats_key).expand(self.num_envs).to(self.device)
        context_dims=64
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, dim=0).to(self.device),
            "true_context": UnboundedContinuousTensorSpec((1,context_dims), device=self.device),
            "pred_context": UnboundedContinuousTensorSpec((1,context_dims), device=self.device),
        }).expand(self.num_envs)

        self.observation_spec["info"] = info_spec
        self.observation_spec["stats"] = stats_spec
        self.info = info_spec.zero()
        self.stats = stats_spec.zero()

        self.observation_h = self.observation_spec[("agents", "observation_h")].zero()
        

    def _reset_idx(self, env_ids: torch.Tensor,judge:bool=False):
        if judge:
            train_judge=False
        else:
            train_judge=True
        self.training=train_judge

        payload_mass_ratio = self.payload_mass_dist.sample(env_ids.shape+(1,))
        self.drone._reset_idx(env_ids, train_judge,self.has_payload, payload_mass_ratio)

        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        # randomly flip the direction
        traj_w = self.traj_w_dist.sample(env_ids.shape) 
        traj_w = traj_w * torch.randn_like(traj_w).sign()
        self.traj_w[env_ids] = traj_w

        t0 = torch.full((len(env_ids),), self.traj_t0, device=self.device)
        pos_0 = lemniscate(t0, self.traj_c[env_ids]) + self.origin
        pos_1 = lemniscate(t0+self.dt*traj_w, self.traj_c[env_ids]) + self.origin
        traj_heading = normalize(pos_1 - pos_0)
        rpy = torch.stack([
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device),
            torch.arctan2(traj_heading[:, 1], traj_heading[:, 0])
        ], dim=-1)
        rot = euler_to_quaternion(rpy).unsqueeze(1)
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        self.drone.set_world_poses(
            pos_0 + self.envs_positions[env_ids], rot, env_ids
        )
        self.drone.set_velocities(vel, env_ids)

        if self.has_payload:
            # TODO@btx0424: workout a better way 
            payload_z = self.payload_z_dist.sample(env_ids.shape)
            joint_indices = torch.tensor([self.drone._view._dof_indices["PrismaticJoint"]], device=self.device)
            self.drone._view.set_joint_positions(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_position_targets(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            
            self.drone._view.set_joint_velocities(
                torch.zeros(len(env_ids), 1, device=self.device), 
                env_indices=env_ids, joint_indices=joint_indices)
            
            payload_mass = payload_mass_ratio * self.drone.masses[env_ids]
            self.payload.set_masses(payload_mass, env_indices=env_ids)

        self.stats[env_ids] = 0.
        self.info["prev_action"][env_ids] = 0.
        self.observation_h[env_ids] = 0.

        if self._should_render(0) and (env_ids == self.central_env_idx).any() :
            # visualize the trajectory
            self.draw.clear_lines()

            traj_vis = self._compute_traj(self.max_episode_length, self.central_env_idx.unsqueeze(0))[0]
            traj_vis = traj_vis + self.envs_positions[self.central_env_idx]
            point_list_0 = traj_vis[:-1].tolist()
            point_list_1 = traj_vis[1:].tolist()
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [1 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)
        self.info["prev_action"][:] = actions
        if not self.training :
            self.info["pred_context"][:] = tensordict['context']
            self.info["true_context"][:] = tensordict['context_target']

    def force_reset(self):
        env_ids = torch.arange(0,self.num_envs,1,device='cuda:0')
        if self.has_payload:
            payload_mass_ratio = self.payload_mass_dist.sample(env_ids.shape+(1,))
            self.drone._reset_idx(env_ids, False,self.has_payload,payload_mass_ratio)
            payload_z = self.payload_z_dist_eval.sample(env_ids.shape)
            joint_indices = torch.tensor([self.drone._view._dof_indices["PrismaticJoint"]], device=self.device)
            self.drone._view.set_joint_positions(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_position_targets(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_velocities(
                torch.zeros(len(env_ids), 1, device=self.device), 
                env_indices=env_ids, joint_indices=joint_indices)
            payload_mass = payload_mass_ratio * self.drone.masses[env_ids]
            self.payload.set_masses(payload_mass, env_indices=env_ids)
        else:
            self.drone._reset_idx(env_ids, False, self.has_payload)     

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]

        self.ref_pos[:] = self._compute_traj(self.future_traj_steps, step_size=5)
        self.ref_heading[:] = normalize(self.ref_pos[:, 1, :2] - self.ref_pos[:, 0, :2])

        self.rpos = self.ref_pos - self.root_state[..., :3]
        obs = [
            self.rpos.flatten(1).unsqueeze(1),
            self.ref_heading.unsqueeze(1) - normalize(self.drone.heading[..., :2]),
            self.root_state[..., 3:],
            self.info["prev_action"]
        ]
        obs_h = [
            self.root_state[..., :],
            self.info["prev_action"]   
        ]
        if self.time_encoding:
            obs.append(self.time_encoding.encode(self.progress_buf).unsqueeze(1))

        obs = torch.cat(obs, dim=-1)
        obs_h = torch.cat(obs_h, dim=-1)
        self.observation_h[..., -1] = obs_h

        return TensorDict({
            "agents": {
                "observation": obs,
                # "observation": self.observation_h[..., -1],
                "observation_h": self.observation_h,
                "intrinsics": self.drone.intrinsics
            },
            "stats": self.stats,
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        pos_error = torch.norm(self.rpos[:, [0]], dim=-1)
        heading_alignment = (
            self.ref_heading.unsqueeze(1) * normalize(self.drone.heading[..., :2])
        ).sum(-1)
        
        reward_pose = torch.exp(-self.reward_distance_scale * pos_error)
        reward_heading = heading_alignment
    
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        reward = (
            reward_pose 
            + reward_pose * (reward_heading) 
            + reward_effort
            + reward_action_smoothness
        )
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        self.stats["tracking_error"].add_(pos_error)
        self.stats["heading_alignment"].add_(heading_alignment)
        if self.training:
            self.stats["return"] += reward
        else:
            '''if self.progress_buf[0] == 1000 :
                self.force_reset()
            if self.progress_buf[0] == 1 :
                self.force_reset()'''
            #mse_loss=torch.nn.MSELoss(reduction='none')
            '''for i in range(self.num_envs):
                mse_key=f"mse_{i}"
                mse=mse_loss(self.info['pred_context'][i,:,:].squeeze(0),self.info['true_context'][i,:,:].squeeze(0)).mean(0).item()
                self.stats[mse_key] = mse*torch.ones_like(reward,device=self.device_name)'''
            #self.stats["mse"] = self.info['pred_context'][0,0,1].item()*torch.ones_like(reward,device=self.device_name)
            reward_weight=torch.where((self.progress_buf > self.warmup_phase_steps * torch.ones_like(self.progress_buf)), 1, 0)[0].item()
            self.stats["return"] += reward_weight * reward
            range_nums_one = ( ( pos_error < 0.1 ) == True ).sum().item()
            range_nums_two = ( ( pos_error < 0.3 ) == True ).sum().item()
            range_nums_three = ( ( pos_error < 1 ) == True ).sum().item()
            range_nums_four = ( ( pos_error < 5 ) == True ).sum().item()
            range_nums_five = ( (pos_error >= 5) == True ).sum().item() 
            self.stats["dist_one"] = range_nums_one / self.num_envs * torch.ones([self.num_envs,1],device=self.device_name)
            self.stats["dist_two"] = range_nums_two / self.num_envs * torch.ones([self.num_envs,1],device=self.device_name)
            self.stats["dist_three"] = range_nums_three / self.num_envs * torch.ones([self.num_envs,1],device=self.device_name)
            self.stats["dist_four"] = range_nums_four / self.num_envs * torch.ones([self.num_envs,1],device=self.device_name)
            self.stats["dist_five"] = range_nums_five / self.num_envs * torch.ones([self.num_envs,1],device=self.device_name)
                        
            '''for i in range(self.num_envs):
                title=f"sample_{i}"            
                sample=self.info['pred_context'][i,:,:]
                sample_data=sample.data.cpu()
                with open(title,"a") as f:
                    np.savetxt(f,sample_data,delimiter=',',fmt="%f")'''

        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)




        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        truncated_eval = (self.progress_buf >= self.max_episode_length - 1).unsqueeze(-1)
        done = (self.drone.pos[..., 2] < 0.1) | (pos_error > self.reset_thres)
        if self.training:
            done = done | truncated
        else:
            done = truncated_eval
        
        self.observation_h[..., :-1] = self.observation_h[..., 1:]
        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": done,
                "truncated": truncated,
            },
            self.batch_size,
        )
    
    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1.):
        if env_ids is None:
            env_ids = ...
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        t = self.traj_t0 + scale_time(self.traj_w[env_ids].unsqueeze(1) * t * self.dt)
        traj_rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, t.shape[1], 4)
        
        ref_pos = vmap(lemniscate)(t, self.traj_c[env_ids])
        ref_pos = vmap(torch_utils.quat_rotate)(traj_rot, ref_pos) * self.traj_scale[env_ids].unsqueeze(1)

        return self.origin + ref_pos
    
    def check_nan(self,input):
        for i in range(self.num_envs):
            if math.isnan(input[i][0].item()):
                input[i][0]=0
        return input 



