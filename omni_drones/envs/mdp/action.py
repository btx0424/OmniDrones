import torch
from .mdp_term import MDPTerm

from omni_drones.robots.multirotor import Multirotor, Rotor

from collections import OrderedDict
from typing import Tuple

class ActionFunc(MDPTerm):
    
    action_shape: torch.Size

    @property
    def action_shape(self):
        raise NotImplementedError
    
    def apply_action(self, action: torch.Tensor):
        pass


class ActionGroup:
    def __init__(self, action_funcs: OrderedDict[str, ActionFunc]):
        assert isinstance(action_funcs, OrderedDict), "ActionGroup requires an OrderedDict of ActionFuncs."
        self.action_funcs = action_funcs
        action_shapes = {key: func.action_shape for key, func in self.action_funcs.items()}
        
        try:
            self.action_shape = torch.cat([torch.zeros(shape) for shape in action_shapes.values()], dim=-1).shape
            self.action_split = [shape[-1] for shape in action_shapes.values()]
        except Exception:
            raise ValueError(f"Incompatible action shapes: {action_shapes}")
    
    def apply_action(self, action: torch.Tensor):
        action_split = torch.split(action, self.action_split, dim=-1)
        for action_func, action in zip(self.action_funcs.values(), action_split):
            action_func.apply_action(action)


class RotorCommand(ActionFunc):
    def __init__(self, env, asset_name: str, actuator_name: str):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]
        self.rotor: Rotor = self.asset.actuators[actuator_name]

    @property
    def action_shape(self):
        return self.rotor.shape
    
    def apply_action(self, action: torch.Tensor):
        self.rotor.throttle_target[:] = ((action + 1.) / 2.).clamp(0., 1.).sqrt()


from omni_drones.utils.torch import (
    quat_mul,
    quat_rotate_inverse,
    quat_rotate,
    normalize, 
    quaternion_to_rotation_matrix,
    quaternion_to_euler,
    axis_angle_to_quaternion,
    axis_angle_to_matrix,
    manual_batch
    
)

class LeePositionController(ActionFunc):

    def __init__(
        self, 
        env: "IsaacEnv", 
        asset_name: str,
        pos_gain: Tuple[float, float, float],
        vel_gain: Tuple[float, float, float],
        ang_rate_gain: Tuple[float, float, float],
        attitute_gain: Tuple[float, float, float],
        inertia: Tuple[float, float, float], # TODO: calculate inertia
        pos: bool = True,
        vel: bool = False,
        yaw: bool = True,
    ):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]
        self.rotor: Rotor = self.asset.actuators["rotor"]

        self.action_dim = 0
        for i, dim in zip([pos, vel, yaw], [3, 3, 1]):
            self.action_dim += dim * i

        rotor_pos_w = self.asset._data.body_pos_w[0, self.rotor.body_ids, :] 
        rotor_pos_b = quat_rotate_inverse(
            self.asset._data.root_quat_w[0].unsqueeze(0),
            rotor_pos_w - self.asset._data.root_pos_w[0].unsqueeze(0)
        )

        arm_lengths = rotor_pos_b.norm(dim=-1)
        rotor_angles = torch.atan2(rotor_pos_b[..., 1], rotor_pos_b[..., 0])
        
        def get_template(tensor):
            return tensor.flatten(0, -2)[0]

        rotor_direction = get_template(self.rotor.rotor_direction)
        moment_to_force = get_template(self.rotor.km_normalized / self.rotor.kf_normalized)

        print(f'[INFO]: arm_lengths: {arm_lengths.tolist()}')
        print(f'[INFO]: rotor_angles: {rotor_angles.tolist()}')
        print(f'[INFO]: rotor_direction: {rotor_direction.tolist()}')

        with torch.device(self.device):
            self.pos_gain = torch.as_tensor(pos_gain)
            self.vel_gain = torch.as_tensor(vel_gain)
            self.gravity = torch.tensor([0., 0., 9.8])

            I = torch.as_tensor([*inertia, 1]).diag_embed()
            A = torch.stack(
                [
                    torch.sin(rotor_angles) * arm_lengths,
                    -torch.cos(rotor_angles) * arm_lengths,
                    -rotor_direction * moment_to_force,
                    torch.ones(self.rotor.shape[-1])
                ]
            )
            self.mixer = A.T @ (A @ A.T).inverse() @ I
            self.ang_rate_gain = torch.as_tensor(ang_rate_gain) @ I[:3, :3].inverse()
            self.attitute_gain = torch.as_tensor(attitute_gain) @ I[:3, :3].inverse()
        
        self.max_thrusts = get_template(self.rotor.kf_normalized)
        self.mass = get_template(
            self.asset.root_physx_view
            .get_masses()
            .sum(-1, keepdim=True)
            .to(self.device)
        )
        
    @property
    def action_shape(self):
        return torch.Size([*self.asset.shape, self.action_dim])
    
    def apply_action(self, action: torch.Tensor):
        batch_shape = action.shape[:-1]
        action = action.reshape(-1, 4)
        target_throttle = self._compute(action)
        self.rotor.throttle_target[:] = target_throttle.reshape(*batch_shape, 4)

    def _compute(self, action: torch.Tensor):
        pos_error, target_yaw = action.split([3, 1], dim=-1)
        vel_error = self.asset._data.root_lin_vel_w - 0.

        acc = (
            - pos_error * self.pos_gain
            + vel_error * self.vel_gain
            - self.gravity
        )

        R = quaternion_to_rotation_matrix(self.asset._data.root_quat_w)
        
        b1_des = torch.cat([
            torch.cos(target_yaw), 
            torch.sin(target_yaw), 
            torch.zeros_like(target_yaw)
        ],dim=-1)

        b3_des = -normalize(acc)
        b2_des = normalize(torch.cross(b3_des, b1_des, 1))
        R_des = torch.stack([
            b2_des.cross(b3_des, 1), 
            b2_des, 
            b3_des
        ], dim=-1)
        ang_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R) 
            - torch.bmm(R.transpose(-2, -1), R_des)
        )
        ang_error = torch.stack([
            ang_error_matrix[:, 2, 1], 
            ang_error_matrix[:, 0, 2], 
            ang_error_matrix[:, 1, 0]
        ],dim=-1)
        ang_vel = self.asset._data.root_ang_vel_b
        ang_rate_err = ang_vel
        ang_acc = (
            - ang_error * self.attitute_gain
            - ang_rate_err * self.ang_rate_gain
            + torch.cross(ang_vel, ang_vel, 1)
        )
        thrust = (-self.mass * (acc * R[:, :, 2]).sum(-1, True))
        ang_acc_thrust = torch.cat([ang_acc, thrust], dim=-1)
        
        target_thrusts = (self.mixer @ ang_acc_thrust.T).T
        target_throttle = target_thrusts / self.max_thrusts
        return target_throttle

    def debug_vis(self):
        rotor_pos_w = self.asset.data.body_pos_w[..., self.rotor.body_ids, :].flatten(0, -2)
        rotor_quat_w = self.asset.data.body_quat_w[..., self.rotor.body_ids, :].flatten(0, -2)
        forces = torch.zeros(self.rotor.shape + (3,), device=self.device)
        forces[..., 2] = self.rotor.throttle
        self.env.debug_draw.vector(
            rotor_pos_w,
            quat_rotate(rotor_quat_w, forces.flatten(0, -2)),
        )