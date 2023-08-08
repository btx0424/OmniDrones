import torch
import torch.nn as nn
from tensordict import TensorDict

from omni_drones.utils.torch import (
    quat_mul,
    normalize, 
    quaternion_to_rotation_matrix,
    axis_angle_to_quaternion,
    axis_angle_to_matrix
)
import yaml
import os.path as osp


def compute_parameters(
    rotor_config,
    inertia_matrix,
):
    rotor_angles = torch.as_tensor(rotor_config["rotor_angles"])
    arm_lengths = torch.as_tensor(rotor_config["arm_lengths"])
    force_constants = torch.as_tensor(rotor_config["force_constants"])
    moment_constants = torch.as_tensor(rotor_config["moment_constants"])
    directions = torch.as_tensor(rotor_config["directions"])
    max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])
    A = torch.stack(
        [
            torch.sin(rotor_angles) * arm_lengths,
            -torch.cos(rotor_angles) * arm_lengths,
            -directions * moment_constants / force_constants,
            torch.ones_like(rotor_angles),
        ]
    )
    mixer = A.T @ (A @ A.T).inverse() @ inertia_matrix

    return mixer

class LeePositionController(nn.Module):
    """
    Computes rotor commands for the given control target using the controller
    described in https://arxiv.org/abs/1003.2005.

    Inputs:
        * root_state: tensor of shape (13,) containing position, rotation (in quaternion),
        linear velocity, and angular velocity.
        * control_target: tensor of shape (7,) contining target position, linear velocity,
        and yaw angle.
    
    Outputs:
        * cmd: tensor of shape (num_rotors,) containing the computed rotor commands.
        * controller_state: empty dict.
    """
    def __init__(
        self, 
        g: float, 
        uav_params,
    ) -> None:
        super().__init__()
        controller_param_path = osp.join(
            osp.dirname(__file__), "cfg", f"lee_controller_{uav_params['name']}.yaml"
        )
        with open(controller_param_path, "r") as f:
            controller_params = yaml.safe_load(f)
        
        self.pos_gain = nn.Parameter(torch.as_tensor(controller_params["position_gain"]).float())
        self.vel_gain = nn.Parameter(torch.as_tensor(controller_params["velocity_gain"]).float())
        self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        self.g = nn.Parameter(torch.tensor([0.0, 0.0, g]).abs())

        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]

        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )
        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        self.attitute_gain = nn.Parameter(
            torch.as_tensor(controller_params["attitude_gain"]).float() @ I[:3, :3].inverse()
        )
        self.ang_rate_gain = nn.Parameter(
            torch.as_tensor(controller_params["angular_rate_gain"]).float() @ I[:3, :3].inverse()
        )
        self.requires_grad_(False)

    def forward(self, root_state: torch.Tensor, control_target: torch.Tensor):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        target_pos, target_vel, target_yaw = torch.split(control_target, [3, 3, 1], dim=-1)

        pos_error = target_pos - pos
        vel_error = target_vel - vel

        acc = (
            pos_error * self.pos_gain 
            + vel_error * self.vel_gain + self.g
        )
        R = quaternion_to_rotation_matrix(rot)
        b1_des = torch.cat(
            [
                torch.cos(target_yaw), 
                torch.sin(target_yaw), 
                torch.zeros_like(target_yaw)
            ],
        dim=-1)
        b3_des = acc / torch.norm(acc, dim=-1, keepdim=True)
        b2_des = normalize(torch.cross(b3_des, b1_des))
        R_des = torch.stack([b2_des.cross(b3_des), b2_des, b3_des], dim=-1)
        ang_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R) 
            - torch.bmm(R.transpose(-2, -1), R_des)
        )
        ang_error = torch.stack(
            [
                ang_error_matrix[:, 2, 1], 
                ang_error_matrix[:, 0, 2], 
                ang_error_matrix[:, 1, 0]
            ],
            dim=-1,
        )
        ang_rate_err = ang_vel
        ang_acc = (
            - ang_error * self.attitute_gain
            - ang_rate_err * self.ang_rate_gain
            + torch.cross(ang_vel, ang_vel)
        )
        thrust = (self.mass * (acc * R[:, :, 2]).sum(-1, True))
        ang_acc_thrust = torch.cat([ang_acc, thrust], dim=-1)
        cmd = (self.mixer @ ang_acc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        return cmd


class AttitudeController(nn.Module):
    r"""
    
    """
    def __init__(self, g, uav_params):
        super().__init__()
        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        self.g = nn.Parameter(torch.tensor(g))
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )

        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        self.gain_attitude = nn.Parameter(
            torch.tensor([3., 3., 0.035]) @ I[:3, :3].inverse()
        )
        self.gain_angular_rate = nn.Parameter(
            torch.tensor([0.52, 0.52, 0.025]) @ I[:3, :3].inverse()
        )


    def forward(self, root_state: torch.Tensor, control_target: torch.Tensor):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3])
        device = pos.device

        R = quaternion_to_rotation_matrix(rot)
        ref_roll, ref_pitch, ref_yaw_rates, ref_thrust = control_target
        yaw = torch.atan2(R[1, 0], R[0, 0])
        yaw = axis_angle_to_matrix(ref_yaw_rates, torch.tensor([0., 0., 1.], device=device))
        roll = axis_angle_to_matrix(ref_roll, torch.tensor([1., 0., 0.], device=device))
        pitch = axis_angle_to_matrix(ref_pitch, torch.tensor([0., 1., 0.], device=device))
        R_des = yaw @ roll @ pitch
        angle_error_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)

        angle_error = torch.stack(
            [angle_error_matrix[2, 1], angle_error_matrix[0, 2], torch.zeros_like(ref_roll)],
            dim=-1,
        )
        angular_rate_des = torch.zeros_like(ang_vel)
        angular_rate_des[2] = ref_yaw_rates
        angular_rate_error = ang_vel - R_des.T @ R @ angular_rate_des

        angular_acc = (
            - angle_error * self.gain_attitude 
            - angular_rate_error * self.gain_angular_rate 
            + torch.cross(ang_vel, ang_vel)
        )
        ref_thrust = (ref_thrust * self.mass * self.g).unsqueeze(-1)
        angular_acc_thrust = torch.cat([angular_acc, ref_thrust])
        cmd = self.mixer @ angular_acc_thrust
        cmd = (cmd / self.max_thrusts) * 2 - 1
        return cmd
