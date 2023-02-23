import torch
import torch.nn as nn
from tensordict import TensorDict
import omni.isaac.core.utils.torch as torch_utils
from .utils import (
    quaternion_to_euler,
    quaternion_to_rotation_matrix, 
    normalize
)

class PositionController(nn.Module):
    def __init__(self,
        g: float, uav_params
    ) -> None:
        super().__init__()
        self.pos_gain = nn.Parameter(torch.tensor([6., 6., 6.]))
        self.vel_gain = nn.Parameter(torch.tensor([4.7, 4.7, 4.7]))
        self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        self.g = nn.Parameter(torch.tensor([0., 0., g]).abs())

        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]

        rotor_angles = torch.as_tensor(rotor_config["rotor_angles"])
        arm_lengths = torch.as_tensor(rotor_config["arm_lengths"])
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        moment_constants = torch.as_tensor(rotor_config["moment_constants"])
        directions = torch.as_tensor(rotor_config["directions"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        A = torch.stack([
            torch.sin(rotor_angles) * arm_lengths,
            -torch.cos(rotor_angles) * arm_lengths,
            -directions * moment_constants / force_constants,
            torch.ones_like(rotor_angles)
        ])
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1]))
        
        self.mixer = nn.Parameter(A.T @ (A @ A.T).inverse() @ I)
        self.attitute_gain = nn.Parameter(torch.tensor([3, 3, 0.15]) @ I[:3, :3].inverse())
        self.ang_rate_gain = nn.Parameter(torch.tensor([0.52, 0.52, 0.18]) @ I[:3, :3].inverse())
        
        print(A)
        print(self.mixer)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self,
        root_state: torch.Tensor,
        control_target: torch.Tensor,
        controller_state: TensorDict
    ):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3])
        target_pos, target_vel, target_yaw = torch.split(control_target, [3, 3, 1])

        pos_error = target_pos - pos 
        vel_error = target_vel - vel

        acc = (
            pos_error * self.pos_gain
            + vel_error * self.vel_gain
            + self.g
        )
        R = quaternion_to_rotation_matrix(rot)
        b1_des = torch.cat([
            torch.cos(target_yaw), 
            torch.sin(target_yaw),
            torch.zeros_like(target_yaw)
        ])
        b3_des = acc / torch.norm(acc, dim=-1) 
        b2_des = normalize(torch.cross(b3_des, b1_des))
        R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)
        ang_error_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
        ang_error = torch.stack([
            ang_error_matrix[2, 1], 
            ang_error_matrix[0, 2], 
            ang_error_matrix[1, 0]
        ], dim=-1)
        ang_rate_err = ang_vel 
        ang_acc = (
            - ang_error * self.attitute_gain
            - ang_rate_err * self.ang_rate_gain
            + torch.cross(ang_vel, ang_vel)
        )
        thrust = (self.mass * (acc @ R[:, 2])).unsqueeze(0)
        ang_acc_thrust = torch.cat([ang_acc, thrust])
        cmd = self.mixer @ ang_acc_thrust
        cmd = (cmd / self.max_thrusts) * 2 - 1
        return cmd, controller_state

