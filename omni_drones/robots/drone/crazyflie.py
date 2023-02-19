import torch
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH
from omni_drones.actuators.damped_motor import NoisyDampedMotor

class Crazyflie(MultirotorBase):

    usd_path: str = get_assets_root_path() + "/Isaac/Robots/Crazyflie/cf2x.usd"
    
    mass = 0.028
    thrust_to_weight = 2.25
    num_rotors = 4
    max_rot_vel = 433.3

    def initialize(self):
        super(MultirotorBase, self).initialize()
        self.base_link = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/{self.name}_*/body",
            name="base_link"
        )
        self.base_link.initialize()
        self.rotors = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/{self.name}_*/m[1-4]_prop",
            name="rotors"
        )
        self.rotors.initialize()
        self.tg = NoisyDampedMotor(
            {
                "max_thrust": 9.81 * self.mass * self.thrust_to_weight / self.num_rotors,
                "max_rot_vel": self.max_rot_vel,
                "rot_directions": self.rot_directions,
            },
            self.base_link,
            self.articulations,
            self.rotors,
            shape=[-1, self._count, self.num_rotors]
        )