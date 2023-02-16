import torch
import abc
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.simulation_context import SimulationContext

class Actuator(abc.ABC):

    def __init__(
        self, 
        cfg,
        articulation_view: ArticulationView,
        rigid_view: RigidPrimView,
        shape: torch.Size,
    ):
        self.cfg = cfg

        shape = torch.Size(shape)
        self.shape = torch.arange(rigid_view.count).reshape(shape).shape
        
        self.rigid_view = rigid_view
        self.articulation_view = articulation_view

        self.dt = SimulationContext._instance.get_physics_dt()
        self.device = SimulationContext._instance._device

    @abc.abstractmethod
    def apply_action(self, actions: torch.Tensor):
        raise NotImplementedError

