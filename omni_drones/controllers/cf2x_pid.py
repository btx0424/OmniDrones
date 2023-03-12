import torch
import torch.nn as nn
from tensordict import TensorDict

from .utils import normalize, quaternion_to_euler, quaternion_to_rotation_matrix


class DSLPIDControl:
    def __init__(self, size, gravity, dt, device="cpu") -> None:
        self.device = device
        self.P_COEFF_FOR = torch.tensor([0.4, 0.4, 1.25], device=device)
        self.I_COEFF_FOR = torch.tensor([0.05, 0.05, 0.05], device=device)
        self.D_COEFF_FOR = torch.tensor([0.2, 0.2, 0.5], device=device)
        self.P_COEFF_TOR = torch.tensor([70000.0, 70000.0, 60000.0], device=device)
        self.I_COEFF_TOR = torch.tensor([0.0, 0.0, 500.0], device=device)
        self.D_COEFF_TOR = torch.tensor([20000.0, 20000.0, 12000.0], device=device)
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000.0
        self.MAX_PWM = 65535.0
        self.GRAVITY = torch.tensor([0, 0, abs(gravity) * 0.027], device=device)
        self.MIXER_MATRIX = torch.tensor(
            [
                # [ .5, -.5, -1], # 0
                # [ .5,  .5,  1], # 1
                # [-.5,  .5, -1], # 2
                # [-.5, -.5,  1], # 3
                [-0.5, -0.5, 1],  # 3
                [-0.5, 0.5, -1],  # 2
                [0.5, 0.5, 1],  # 1
                [0.5, -0.5, -1],  # 0
            ],
            device=device,
        )
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

    def compute_control(
        self, cur_state, target_pos, target_rpy, target_vel, target_rpy_rates
    ):
        (
            rpms,
            self.last_rpy[:],
            self.integral_pos_e[:],
            self.integral_rpy_e[:],
        ) = compute_control(
            self.control_timestep,
            cur_state,
            target_pos,
            target_rpy,
            target_vel,
            target_rpy_rates,
            self.last_rpy,
            self.integral_rpy_e,
            self.integral_pos_e,
            self.MIXER_MATRIX,
            self.PWM2RPM_SCALE,
            self.PWM2RPM_CONST,
            self.P_COEFF_FOR,
            self.I_COEFF_FOR,
            self.D_COEFF_FOR,
            self.P_COEFF_TOR,
            self.I_COEFF_TOR,
            self.D_COEFF_TOR,
            self.GRAVITY,
            self.MIN_PWM,
            self.MAX_PWM,
            self.KF,
        )
        rpms.nan_to_num_(0.0)
        return rpms


from torch import Tensor

EPS = 1e-6
# @torch.jit.script
def compute_control(
    control_timestep: float,
    cur_state: Tensor,
    target_pos: Tensor,
    target_rpy: Tensor,
    target_vel: Tensor,
    target_rpy_rates: Tensor,
    last_rpy: Tensor,
    integral_rpy_e: Tensor,
    integral_pos_e: Tensor,
    MIXER_MATRIX: Tensor,
    PWM2RPM_SCALE: Tensor,
    PWM2RPM_CONST: Tensor,
    P_COEFF_FOR: Tensor,
    I_COEFF_FOR: Tensor,
    D_COEFF_FOR: Tensor,
    P_COEFF_TOR: Tensor,
    I_COEFF_TOR: Tensor,
    D_COEFF_TOR: Tensor,
    GRAVITY: Tensor,
    MIN_PWM: float,
    MAX_PWM: float,
    KF: float,
):
    # position control
    cur_pos, cur_quat, cur_vel, cur_angvel = torch.split(
        cur_state, [3, 4, 3, 3], dim=cur_state.dim() - 1
    )
    pos_e = target_pos - cur_pos
    vel_e = target_vel - cur_vel

    cur_rotation = quaternion_to_rotation_matrix(cur_quat)  # (*, 3, 3)

    integral_pos_e = integral_pos_e + pos_e * control_timestep
    integral_pos_e = torch.clip(integral_pos_e, -2.0, 2.0)
    integral_pos_e[..., 2] = torch.clip(integral_pos_e[..., 2], -0.15, 0.15)

    target_thrust = (
        P_COEFF_FOR * pos_e
        + I_COEFF_FOR * integral_pos_e
        + D_COEFF_FOR * vel_e
        + GRAVITY
    )

    scalar_thrust = (
        (target_thrust * cur_rotation[..., :, 2]).sum(-1, keepdim=True).clamp_min(0)
    )
    thrust = (torch.sqrt(scalar_thrust / (4 * KF)) - PWM2RPM_CONST) / PWM2RPM_SCALE
    target_x_c = torch.stack(
        [
            torch.cos(target_rpy[..., 2]),
            torch.sin(target_rpy[..., 2]),
            torch.zeros_like(target_rpy[..., 2]),
        ],
        dim=-1,
    )
    target_z_ax = target_thrust / (
        torch.norm(target_thrust, dim=-1, keepdim=True) + EPS
    )
    target_y_ax = torch.cross(target_z_ax, target_x_c, dim=-1)
    target_y_ax /= torch.norm(target_y_ax, dim=-1, keepdim=True) + EPS
    target_x_ax = torch.cross(target_y_ax, target_z_ax, dim=-1)
    target_rotation = torch.stack([target_x_ax, target_y_ax, target_z_ax], dim=-1)

    # attitude control
    cur_rpy = quaternion_to_euler(cur_quat)

    rot_matrix_e = torch.matmul(
        target_rotation.transpose(-1, -2), cur_rotation
    ) - torch.matmul(cur_rotation.transpose(-1, -2), target_rotation)

    rot_e = torch.stack(
        [rot_matrix_e[..., 2, 1], rot_matrix_e[..., 0, 2], rot_matrix_e[..., 1, 0]],
        dim=-1,
    )
    rpy_rates_e = target_rpy_rates - (cur_rpy - last_rpy) / control_timestep

    integral_rpy_e = integral_rpy_e - rot_e * control_timestep
    integral_rpy_e.clamp_(-1500.0, 1500.0)
    integral_rpy_e[..., :2].clamp_(-1.0, 1.0)
    #### PID target torques ####################################
    target_torques = (
        -P_COEFF_TOR * rot_e + D_COEFF_TOR * rpy_rates_e + I_COEFF_TOR * integral_rpy_e
    )

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
