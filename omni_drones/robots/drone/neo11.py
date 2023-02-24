import torch
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni.isaac.core.prims import RigidPrimView

from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH

class Neo11(MultirotorBase):

    usd_path: str = ASSET_PATH + "/usd/neo11.usd"
    param_path: str = ASSET_PATH + "/usd/neo11.yaml"

