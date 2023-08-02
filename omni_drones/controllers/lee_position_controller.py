import torch
import torch.nn as nn
from tensordict import TensorDict

from .utils import normalize, quaternion_to_rotation_matrix
import yaml
import os.path as osp

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
        dt: float, 
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

        rotor_angles = torch.as_tensor(rotor_config["rotor_angles"])
        arm_lengths = torch.as_tensor(rotor_config["arm_lengths"])
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        moment_constants = torch.as_tensor(rotor_config["moment_constants"])
        directions = torch.as_tensor(rotor_config["directions"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        A = torch.stack(
            [
                torch.sin(rotor_angles) * arm_lengths,
                -torch.cos(rotor_angles) * arm_lengths,
                -directions * moment_constants / force_constants,
                torch.ones_like(rotor_angles),
            ]
        )
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )

        self.mixer = nn.Parameter(A.T @ (A @ A.T).inverse() @ I)
        self.attitute_gain = nn.Parameter(
            torch.as_tensor(controller_params["attitude_gain"]).float() @ I[:3, :3].inverse()
        )
        self.ang_rate_gain = nn.Parameter(
            torch.as_tensor(controller_params["angular_rate_gain"]).float() @ I[:3, :3].inverse()
        )

        self.requires_grad_(False)

    def forward(
        self,
        root_state: torch.Tensor,
        control_target: torch.Tensor,
        controller_state: TensorDict,
    ):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3])
        target_pos, target_vel, target_yaw = torch.split(control_target, [3, 3, 1])

        pos_error = target_pos - pos
        vel_error = target_vel - vel

        acc = pos_error * self.pos_gain + vel_error * self.vel_gain + self.g
        R = quaternion_to_rotation_matrix(rot)
        b1_des = torch.cat(
            [torch.cos(target_yaw), torch.sin(target_yaw), torch.zeros_like(target_yaw)]
        )
        b3_des = acc / torch.norm(acc, dim=-1)
        b2_des = normalize(torch.cross(b3_des, b1_des))
        R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)
        ang_error_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
        ang_error = torch.stack(
            [ang_error_matrix[2, 1], ang_error_matrix[0, 2], ang_error_matrix[1, 0]],
            dim=-1,
        )
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
