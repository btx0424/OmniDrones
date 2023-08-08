import torch
import torch.nn as nn
from tensordict import TensorDict

from omni_drones.utils.torch import normalize, quaternion_to_euler, quaternion_to_rotation_matrix


class DSLPIDController(nn.Module):
    def __init__(self, dt: float, g: float, uav_params) -> None:
        super().__init__()
        self.P_COEFF_FOR = nn.Parameter(torch.tensor([0.4, 0.4, 1.25]))
        self.I_COEFF_FOR = nn.Parameter(torch.tensor([0.05, 0.05, 0.05]))
        self.D_COEFF_FOR = nn.Parameter(torch.tensor([0.2, 0.2, 0.5]))
        self.P_COEFF_TOR = nn.Parameter(torch.tensor([70000.0, 70000.0, 60000.0]))
        self.I_COEFF_TOR = nn.Parameter(torch.tensor([0.0, 0.0, 500.0]))
        self.D_COEFF_TOR = nn.Parameter(torch.tensor([20000.0, 20000.0, 12000.0]))
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000.0
        self.MAX_PWM = 65535.0
        self.GRAVITY = nn.Parameter(torch.tensor([0, 0, abs(g) * uav_params["mass"]]))
        self.MIXER_MATRIX = nn.Parameter(
            torch.tensor(
                [
                    # [ .5, -.5, -1], # 0
                    # [ .5,  .5,  1], # 1
                    # [-.5,  .5, -1], # 2
                    # [-.5, -.5,  1], # 3
                    [-0.5, -0.5, 1],  # 3
                    [-0.5, 0.5, -1],  # 2
                    [0.5, 0.5, 1],  # 1
                    [0.5, -0.5, -1],  # 0
                ]
            )
        )
        self.KF = 3.16e-10
        self.dt = dt
        self.MAX_RPM = 21714
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        state: torch.Tensor,
        control_target: torch.Tensor,
        controller_state: TensorDict,
    ):
        # parsing input
        pos, quat, vel, angvel = torch.split(state, [3, 4, 3, 3])
        target_pos, target_vel, target_yaw = torch.split(control_target, [3, 3, 1])
        rpy = quaternion_to_euler(quat)
        rot = quaternion_to_rotation_matrix(quat)
        integral_pos_error = controller_state.get(
            "integral_pos_error", torch.zeros_like(pos)
        )
        integral_rpy_error = controller_state.get(
            "integral_rpy_error", torch.zeros_like(rpy)
        )
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
        target_x_c = torch.cat(
            [torch.cos(target_yaw), torch.sin(target_yaw), torch.zeros_like(target_yaw)]
        )
        target_z_ax = normalize(target_thrust)
        target_y_ax = normalize(torch.cross(target_z_ax, target_x_c))
        target_x_ax = torch.cross(target_y_ax, target_z_ax)
        target_rot = torch.stack([target_x_ax, target_y_ax, target_z_ax], dim=-1)
        rot_matrix_error = target_rot.T @ rot - rot.T @ target_rot
        rot_error = torch.stack(
            [rot_matrix_error[2, 1], rot_matrix_error[0, 2], rot_matrix_error[1, 0]]
        )
        rpy_rates_error = -(rpy - last_rpy) / self.dt
        integral_rpy_error = integral_rpy_error - rot_error * self.dt
        target_torque = (
            -self.P_COEFF_TOR * rot_error
            + self.D_COEFF_TOR * rpy_rates_error
            + self.I_COEFF_TOR * integral_rpy_error
        )

        thrust = (
            torch.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST
        ) / self.PWM2RPM_SCALE
        pwm = torch.clip(thrust + self.MIXER_MATRIX @ target_torque, 0, 65535)
        rpms = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        cmd = torch.square(rpms / self.MAX_RPM) * 2 - 1

        controller_state.update(
            {
                "integral_pos_error": integral_pos_error,
                "integral_rpy_error": integral_pos_error,
                "last_rpy": rpy,
            }
        )
        return cmd, controller_state
