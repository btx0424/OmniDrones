import torch
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni.isaac.core.prims import RigidPrimView

from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH

class Neo11(MultirotorBase):

    usd_path: str = ASSET_PATH + "/usd/neo11.usd"

    mass = 3.45
    thrust_to_weight = 2.25
    num_rotors = 6
    max_rot_vel = 1047

