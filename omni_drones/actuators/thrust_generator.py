import torch

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.simulation_context import SimulationContext

class ThrustGenerator:

    def __init__(self, view: RigidPrimView):
        assert view._physics_view is not None
        self.view = view
        self.thrust = torch.zeros(self.view.count, 3, device=self.view._device)
        
    def apply(self, actions: torch.Tensor):
        dt = SimulationContext._instance.get_physics_dt()
        self.thrust += dt * actions
        
        self.view.apply_forces(self.thrust)
