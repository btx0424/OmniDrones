import torch
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni.isaac.core.prims import RigidPrimView

from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH

class Hummingbird(MultirotorBase):

    usd_path: str = ASSET_PATH + "/usd/hummingbird.usd"

    mass = 0.72
    thrust_to_weight = 2.25
    num_rotors = 4
    max_rot_vel = 838

