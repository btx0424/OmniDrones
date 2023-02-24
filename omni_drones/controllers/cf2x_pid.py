import torch
import torch.nn as nn
from tensordict import TensorDict
from .utils import (
    quaternion_to_rotation_matrix,
    quaternion_to_euler,
    normalize
)

class DSLPIDControl:
    
    def __init__(self, 
        size, gravity, dt, device="cpu"
    ) -> None:
        self.device = device
        self.P_COEFF_FOR = torch.tensor([.4, .4, 1.25], device=device)
        self.I_COEFF_FOR = torch.tensor([.05, .05, .05], device=device)
        self.D_COEFF_FOR = torch.tensor([.2, .2, .5], device=device)
        self.P_COEFF_TOR = torch.tensor([70000., 70000., 60000.], device=device)
        self.I_COEFF_TOR = torch.tensor([.0, .0, 500.], device=device)
        self.D_COEFF_TOR = torch.tensor([20000., 20000., 12000.], device=device)
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000.
        self.MAX_PWM = 65535.
        self.GRAVITY = torch.tensor([0, 0, abs(gravity) * 0.027], device=device)
        self.MIXER_MATRIX = torch.tensor([
            # [ .5, -.5, -1], # 0
            # [ .5,  .5,  1], # 1
            # [-.5,  .5, -1], # 2
            # [-.5, -.5,  1], # 3
            [-.5, -.5,  1], # 3
            [-.5,  .5, -1], # 2
            [ .5,  .5,  1], # 1
            [ .5, -.5, -1], # 0
        ], device=device)
        self.KF = 3.16e-10
        self.size = size
        self.control_timestep = dt

        self.last_pos_e = torch.zeros((*size, 3), device=self.device)
        self.integral_pos_e = torch.zeros((*size, 3), device=self.device)
        self.last_rpy = torch.zeros((*size, 3), device=self.device)
        self.last_rpy_e = torch.zeros((*size, 3), device=self.device)
        self.integral_rpy_e = torch.zeros((*size, 3), device=self.device)

    def reset_idx(self, env_ids: torch.Tensor):
        self.last_pos_e[env_ids] = 0
        self.integral_pos_e[env_ids] = 0
        self.last_rpy[env_ids] = 0
        self.last_rpy_e[env_ids] = 0
        self.integral_rpy_e[env_ids] = 0

    def compute_control(self, 
            cur_state, target_pos, target_rpy, target_vel, target_rpy_rates
        ):
        rpms, self.last_rpy[:], self.integral_pos_e[:], self.integral_rpy_e[:] = compute_control(
            self.control_timestep, 
            cur_state,
            target_pos, target_rpy, target_vel, target_rpy_rates, 
            self.last_rpy, self.integral_rpy_e, self.integral_pos_e,
            self.MIXER_MATRIX, self.PWM2RPM_SCALE, self.PWM2RPM_CONST,
            self.P_COEFF_FOR, self.I_COEFF_FOR, self.D_COEFF_FOR,
            self.P_COEFF_TOR, self.I_COEFF_TOR, self.D_COEFF_TOR,
            self.GRAVITY, self.MIN_PWM, self.MAX_PWM, self.KF
        )
        rpms.nan_to_num_(0.)
        return rpms

from torch import Tensor

EPS = 1e-6
# @torch.jit.script
def compute_control(
    control_timestep: float, 
    cur_state: Tensor,
    target_pos: Tensor, target_rpy: Tensor, target_vel: Tensor, target_rpy_rates: Tensor,
    last_rpy: Tensor, integral_rpy_e: Tensor, integral_pos_e: Tensor,
    MIXER_MATRIX: Tensor, PWM2RPM_SCALE: Tensor, PWM2RPM_CONST: Tensor,
    P_COEFF_FOR: Tensor, I_COEFF_FOR: Tensor, D_COEFF_FOR: Tensor,
    P_COEFF_TOR: Tensor, I_COEFF_TOR: Tensor, D_COEFF_TOR: Tensor,
    GRAVITY: Tensor, MIN_PWM: float, MAX_PWM: float, KF: float
):
    # position control
    cur_pos, cur_quat, cur_vel, cur_angvel = torch.split(cur_state, [3, 4, 3, 3], dim=cur_state.dim()-1)
    pos_e = target_pos - cur_pos
    vel_e = target_vel - cur_vel

    cur_rotation = quaternion_to_rotation_matrix(cur_quat) # (*, 3, 3)

    integral_pos_e = integral_pos_e + pos_e*control_timestep
    integral_pos_e = torch.clip(integral_pos_e, -2., 2.)
    integral_pos_e[..., 2] = torch.clip(integral_pos_e[..., 2], -0.15, .15)

    target_thrust = P_COEFF_FOR * pos_e \
        + I_COEFF_FOR * integral_pos_e \
        + D_COEFF_FOR * vel_e \
        + GRAVITY

    scalar_thrust = (target_thrust*cur_rotation[...,:,2]).sum(-1, keepdim=True).clamp_min(0)
    thrust = (torch.sqrt(scalar_thrust / (4 * KF)) - PWM2RPM_CONST) / PWM2RPM_SCALE
    target_x_c = torch.stack([
        torch.cos(target_rpy[..., 2]), torch.sin(target_rpy[..., 2]), torch.zeros_like(target_rpy[..., 2])], dim=-1)
    target_z_ax = target_thrust / (torch.norm(target_thrust, dim=-1, keepdim=True) + EPS)
    target_y_ax = torch.cross(target_z_ax, target_x_c, dim=-1)
    target_y_ax /= (torch.norm(target_y_ax, dim=-1, keepdim=True) + EPS)
    target_x_ax = torch.cross(target_y_ax, target_z_ax, dim=-1)
    target_rotation = torch.stack([target_x_ax, target_y_ax, target_z_ax], dim=-1)
    
    # attitude control
    cur_rpy = quaternion_to_euler(cur_quat)

    rot_matrix_e = torch.matmul(target_rotation.transpose(-1, -2), cur_rotation) - torch.matmul(cur_rotation.transpose(-1, -2), target_rotation)

    rot_e = torch.stack([rot_matrix_e[..., 2, 1], rot_matrix_e[..., 0, 2], rot_matrix_e[..., 1, 0]], dim=-1)
    rpy_rates_e = target_rpy_rates - (cur_rpy - last_rpy) / control_timestep

    integral_rpy_e = integral_rpy_e - rot_e*control_timestep
    integral_rpy_e.clamp_(-1500., 1500.)
    integral_rpy_e[..., :2].clamp_(-1., 1.)
    #### PID target torques ####################################
    target_torques = - P_COEFF_TOR * rot_e \
                    + D_COEFF_TOR * rpy_rates_e \
                    + I_COEFF_TOR * integral_rpy_e
    
    target_torques = torch.clip(target_torques, -3200, 3200)
    pwm = thrust + torch.einsum("ij,...j->...i", MIXER_MATRIX, target_torques)
    pwm = torch.clip(pwm, MIN_PWM, MAX_PWM)
    rpms = PWM2RPM_SCALE * pwm + PWM2RPM_CONST
    # thrust = (torch.sqrt(scalar_thrust / (4 * KF)) - PWM2RPM_CONST) / PWM2RPM_SCALE

    return (
        rpms,
        cur_rpy,
        integral_pos_e,
        integral_rpy_e,
    )

import torch.nn as nn

class PIDController(nn.Module):
    def __init__(self,
        dt, mass, g
    ) -> None:
        super().__init__()
        self.P_COEFF_FOR = nn.Parameter(torch.tensor([.4, .4, 1.25]))
        self.I_COEFF_FOR = nn.Parameter(torch.tensor([.05, .05, .05]))
        self.D_COEFF_FOR = nn.Parameter(torch.tensor([.2, .2, .5]))
        self.P_COEFF_TOR = nn.Parameter(torch.tensor([70000., 70000., 60000.]))
        self.I_COEFF_TOR = nn.Parameter(torch.tensor([.0, .0, 500.]))
        self.D_COEFF_TOR = nn.Parameter(torch.tensor([20000., 20000., 12000.]))
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000.
        self.MAX_PWM = 65535.
        self.GRAVITY = nn.Parameter(torch.tensor([0, 0, abs(g) * mass]))
        self.MIXER_MATRIX = nn.Parameter(torch.tensor([
            # [ .5, -.5, -1], # 0
            # [ .5,  .5,  1], # 1
            # [-.5,  .5, -1], # 2
            # [-.5, -.5,  1], # 3
            [-.5, -.5,  1], # 3
            [-.5,  .5, -1], # 2
            [ .5,  .5,  1], # 1
            [ .5, -.5, -1], # 0
        ]))
        self.KF = 3.16e-10
        self.dt = dt
        self.MAX_RPM = ...
        for p in self.parameters():
            p.requires_grad_(False)

    
    def forward(self, state: torch.Tensor, control_target: torch.Tensor, controller_state: TensorDict):
        # parsing input
        pos, quat, vel, angvel = torch.split(state, [3, 4, 3, 3])
        target_pos, target_vel, target_yaw = torch.split(control_target, [3, 3, 1])
        rpy = quaternion_to_euler(quat)
        rot = quaternion_to_rotation_matrix(quat)
        integral_pos_error = controller_state.get("integral_pos_error", torch.zeros_like(pos))
        integral_rpy_error = controller_state.get("integral_rpy_error", torch.zeros_like(rpy))
        last_rpy = controller_state.get("last_rpy", rpy)

        # position control
        pos_error = target_pos - pos
        vel_error = target_vel - vel
        integral_pos_error = torch.clip(integral_pos_error + pos_error * self.dt, -2, 2)

        target_thrust = (
            self.P_COEFF_FOR * pos_error
            + self.I_COEFF_FOR * integral_pos_error
            + self.D_COEFF_FOR * vel_error
            + self.GRAVITY
        )
        scalar_thrust = target_thrust @ rot[:, 2]

        # attitute control
        target_x_c = torch.cat([torch.cos(target_yaw), torch.sin(target_yaw), torch.zeros_like(target_yaw)])
        target_z_ax = normalize(target_thrust)
        target_y_ax = normalize(torch.cross(target_z_ax, target_x_c))
        target_x_ax = torch.cross(target_y_ax, target_z_ax)
        target_rot = torch.stack([target_x_ax, target_y_ax, target_z_ax], dim=-1)
        rot_matrix_error = target_rot.T @ rot - rot.T @ target_rot
        rot_error = torch.stack([rot_matrix_error[2, 1], rot_matrix_error[0, 2], rot_matrix_error[1, 0]])
        rpy_rates_error = - (rpy - last_rpy) / self.dt
        integral_rpy_error = integral_rpy_error - rot_error * self.dt
        target_torque = (
            -self.P_COEFF_TOR * rot_error
            + self.D_COEFF_TOR * rpy_rates_error
            + self.I_COEFF_TOR * integral_rpy_error
        )

        thrust = (torch.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        pwm = torch.clip(thrust + self.MIXER_MATRIX @ target_torque, 0, 65535)
        rpms = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        cmd = torch.square(rpms / self.MAX_RPM) * 2 - 1
        return cmd, controller_state

