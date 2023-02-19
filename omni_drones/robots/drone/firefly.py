import torch

from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH

class Firefly(MultirotorBase):

    usd_path: str = ASSET_PATH + "/usd/firefly.usd"
    
    mass = 1.5
    thrust_to_weight = 2.25
    num_rotors = 6
    max_rot_vel = 838

