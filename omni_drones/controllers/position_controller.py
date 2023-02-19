import torch
import omni.isaac.core.utils.torch as torch_utils
from .utils import quaternion_to_rotation_matrix

class PositionController:
    def __init__(self) -> None:
        self.pos_gain = ...
        self.vel_gain = ...


def compute(
    pos: torch.Tensor, vel: torch.Tensor,
    target_pos: torch.Tensor, target_vel: torch.Tensor,
    pos_gain: torch.Tensor, vel_gain: torch.Tensor, 
    rot: torch.Tensor, target_yaw: torch.Tensor,
    ang_vel: torch.Tensor, attitute_gain: torch.Tensor,
    mass: torch.Tensor, gravity: torch.Tensor
):
    pos_error = pos - target_pos
    vel_error = vel - target_vel

    acc = (
        pos_error * pos_gain
        + vel_error * vel_gain
    )

    R = quaternion_to_rotation_matrix(rot)
    b1_des = torch.cat([
        torch.cos(target_yaw), 
        torch.sin(target_yaw),
        torch.zeros_like(target_yaw)
    ])
    b3_des = - acc / torch.norm(acc, dim=-1) 
    b2_des = torch.cross(b2_des, b3_des)
    R_des = torch.stack([b1_des, b2_des, b3_des]).T

    ang_error_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
    ang_error = torch.cat([ang_error_matrix[2, 1], ang_error_matrix[0, 2], ang_error_matrix[1, 0]])
    ang_acc = (
        -1 * ang_error * attitute_gain
        + torch.cross(ang_vel, ang_vel)
    )
    thrust = torch_utils.quat_rotate(rot, mass * acc)
    
    return 

