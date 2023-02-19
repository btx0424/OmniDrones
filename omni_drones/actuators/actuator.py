import torch
import abc
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.simulation_context import SimulationContext

class Actuator(abc.ABC):

    def __init__(
        self, 
        cfg,
        base_link: RigidPrimView,
        joints: ArticulationView,
        bodies: RigidPrimView,
        shape: torch.Size,
    ):
        self.cfg = cfg

        shape = torch.Size(shape)
        self.shape = torch.arange(bodies.count).reshape(shape).shape
        
        self.base_link = base_link
        self.bodies = bodies
        self.joints = joints

        self.dt = SimulationContext._instance.get_physics_dt()
        self.device = SimulationContext._instance._device

    @abc.abstractmethod
    def apply_action(self, actions: torch.Tensor):
        raise NotImplementedError

