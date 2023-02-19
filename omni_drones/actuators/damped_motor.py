import torch

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.torch.rotations import quat_axis
from omni_drones.actuators.actuator import Actuator

EPS = 1e-6   # small constant to avoid divisions by 0 and log(0)

class NoisyDampedMotor(Actuator):
    """The motor model in OmniIsaacGymEnvs' Crazyflie task."""
    def __init__(self,
        cfg,
        base_link: RigidPrimView,
        joints: ArticulationView,   
        rotors: RigidPrimView,
        shape: torch.Size,
    ) -> None:
        super().__init__(cfg, base_link, joints, rotors, shape)

        self.max_thrust = torch.as_tensor(cfg["max_thrust"], device=self.device)
        self.max_rot_vel = torch.as_tensor(cfg["max_rot_vel"], device=self.device)
        
        rot_directions = cfg.get("rot_directions", None)
        if rot_directions is None:
            rot_directions = torch.ones(self.shape[-1], device=self.device)
            rot_directions[1::2] = -1
        else:
            rot_directions = torch.as_tensor(rot_directions, device=self.device)
            assert len(rot_directions) == self.shape[-1]
        self.rot_directions = rot_directions

        # parameters for the controller
        self.motor_damp_time_up = 0.15
        self.motor_damp_time_down = 0.15

        # I use the multiplier 4, since 4*T ~ time for a step response to finish, where
        # T is a time constant of the first-order filter
        self.motor_tau_up = 4 * self.dt / (self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4 * self.dt / (self.motor_damp_time_down + EPS)

        self.thrusts = torch.zeros((*self.shape, 3), device=self.device)
        self.thrust_cmds_damp = torch.zeros(*self.shape, device=self.device)
        self.thrust_rot_damp = torch.zeros(*self.shape, device=self.device)
        self.torques = torch.zeros((*self.shape[:-1], 3), device=self.device)

    def apply_action(self, actions: torch.Tensor):
        # clamp to [-1.0, 1.0]
        thrust_cmds = torch.clamp(actions, min=-1.0, max=1.0)
        # scale to [0.0, 1.0]
        thrust_cmds = (thrust_cmds + 1.0) / 2.0
        # filtering the thruster and adding noise
        motor_tau = torch.where(
            thrust_cmds > self.thrust_cmds_damp,
            self.motor_tau_up,
            self.motor_tau_down
        )

        # Since NN commands thrusts we need to convert to rot vel and back
        thrust_rot = thrust_cmds ** 0.5
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
        self.thrust_cmds_damp = self.thrust_rot_damp ** 2

        z_thrusts = self.max_thrust * self.thrust_cmds_damp
        self.thrusts[..., 2] = z_thrusts

        prop_rot = self.thrust_cmds_damp * self.max_rot_vel
        dof_vel = prop_rot * self.rot_directions

        self.joints.set_joint_velocities(dof_vel.reshape(-1, self.joints.num_dof))
        self.bodies.apply_forces(forces=self.thrusts.reshape(-1, 3), is_global=False)
        
        z_torques = (self.thrust_cmds_damp * self.rot_directions).sum(-1)
        self.torques[..., 2] = z_torques * 0.1
        self.base_link.apply_forces_and_torques_at_pos(None, self.torques.reshape(-1, 3), is_global=False)
        
