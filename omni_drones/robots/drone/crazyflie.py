import torch

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from tensordict.nn import make_functional
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.controllers import DSLPIDController
from omni_drones.robots import ASSET_PATH
from omni_drones.robots.drone import MultirotorBase


class Crazyflie(MultirotorBase):

    # usd_path: str = get_assets_root_path() + "/Isaac/Robots/Crazyflie/cf2x.usd"
    usd_path: str = ASSET_PATH + "/usd/cf2x_isaac.usd"
    param_path: str = ASSET_PATH + "/usd/crazyflie.yaml"
    DEFAULT_CONTROLLER = DSLPIDController
