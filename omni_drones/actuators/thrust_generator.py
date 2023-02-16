import torch

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.simulation_context import SimulationContext
from omni_drones.actuators.actuator import Actuator

class ThrustGenerator(Actuator):

    def __init__(
        self, 
        cfg,
        articulation_view: ArticulationView,
        rigid_view: RigidPrimView,
        shape: torch.Size,
    ):
        super().__init__(cfg, articulation_view, rigid_view, shape)

        max_thrust = cfg["max_thrust"]
        self.thrust_action_speed_scale = cfg["thrust_action_speed_scale"]
        self.thrust = torch.zeros(*self.shape, 3, device=self.device)
        self.thrust_lower_limits = -max_thrust * torch.ones(4, device=self.device)
        self.thrust_upper_limits = max_thrust * torch.ones(4, device=self.device)

    def apply_action(self, actions: torch.Tensor):
        self.thrust[..., 2] += self.dt * self.thrust_action_speed_scale * actions
        self.thrust[..., 2] = torch.clamp(
            self.thrust[..., 2],
            self.thrust_lower_limits, 
            self.thrust_upper_limits
        )
        self.rigid_view.apply_forces(self.thrust.flatten(0, -2), is_global=False)

