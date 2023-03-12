import torch

from omni.isaac.core.prims import RigidPrimView
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH


class Hummingbird(MultirotorBase):

    usd_path: str = ASSET_PATH + "/usd/hummingbird.usd"
    param_path: str = ASSET_PATH + "/usd/hummingbird.yaml"
