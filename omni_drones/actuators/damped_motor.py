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
        articulation_view: ArticulationView,   
        rigid_view: RigidPrimView,
        shape: torch.Size,
    ) -> None:
        super().__init__(cfg, articulation_view, rigid_view, shape)

        self.thrust_max = torch.as_tensor(cfg["thrust_max"], device=self.device)
        self.prop_max_rot = torch.as_tensor(cfg["prop_max_rot"], device=self.device)
        
        self.articulation_view = articulation_view
        self.rigid_view = rigid_view

        # parameters for the controller
        self.motor_damp_time_up = 0.15
        self.motor_damp_time_down = 0.15

        # I use the multiplier 4, since 4*T ~ time for a step response to finish, where
        # T is a time constant of the first-order filter
        self.motor_tau_up = 4 * self.dt / (self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4 * self.dt / (self.motor_damp_time_down + EPS)

        self.thrusts = torch.zeros((*self.shape, 3), dtype=torch.float32, device=self.device)
        self.thrust_cmds_damp = torch.zeros(*self.shape, dtype=torch.float32, device=self.device)
        self.thrust_rot_damp = torch.zeros(*self.shape, dtype=torch.float32, device=self.device)

        self.prop_rot_direction = -torch.ones(self.shape[-1], device=self.device)
        self.prop_rot_direction[::2] = 1

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

        z_thrusts = self.thrust_max * self.thrust_cmds_damp
        self.thrusts[..., 2] = z_thrusts

        prop_rot = self.thrust_cmds_damp * self.prop_max_rot
        dof_vel = prop_rot * self.prop_rot_direction

        self.articulation_view.set_joint_velocities(dof_vel.reshape(-1, self.articulation_view.num_dof))
        self.rigid_view.apply_forces(self.thrusts.flatten(0, -2), is_global=False)
        
