import torch

from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH

class Firefly(MultirotorBase):

    usd_path: str = ASSET_PATH + "/usd/firefly.usd"
    param_path: str = ASSET_PATH + "/usd/firefly.yaml"

