import torch
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec
from tensordict.nn import make_functional

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni_drones.robots import ASSET_PATH
from omni_drones.robots.drone import MultirotorBase
from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.controllers import DSLPIDController

class Crazyflie(MultirotorBase):

    usd_path: str = get_assets_root_path() + "/Isaac/Robots/Crazyflie/cf2x.usd"
    param_path: str = ASSET_PATH + "/usd/crazyflie.yaml"
    default_controller = DSLPIDController

    def initialize(self):
        super(MultirotorBase, self).initialize()
        self.base_link = RigidPrimView(
            prim_paths_expr=f"{self.prim_paths_expr}/body", name="base_link")
        self.base_link.initialize()
        self.rotors_view = RigidPrimView(
            prim_paths_expr=f"{self.prim_paths_expr}/m[1-4]_prop", name="rotors")
        self.rotors_view.initialize()
        
        self.rotors = RotorGroup(self.params["rotor_configuration"], dt=self.dt).to(self.device)
        self.rotor_params_and_states = make_functional(self.rotors).expand(self.shape).clone()
        
        self.max_forces = self.rotor_params_and_states["max_forces"]
        self.throttle = self.rotor_params_and_states["throttle"]
        self.KF = self.rotor_params_and_states["KF"]
        self.KM = self.rotor_params_and_states["KM"]
        self.directions = self.rotor_params_and_states["directions"]
        
        self.forces = torch.zeros(*self.shape, self.num_rotors, 3, device=self.device)
        self.torques = torch.zeros(*self.shape, 3, device=self.device)

