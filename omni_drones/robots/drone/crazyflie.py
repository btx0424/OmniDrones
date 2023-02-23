import torch
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH

from math import sqrt
class Crazyflie(MultirotorBase):
    ...
    # usd_path: str = get_assets_root_path() + "/Isaac/Robots/Crazyflie/cf2x.usd"

    # mass = 0.028
    # thrust_to_weight = 2.25
    # num_rotors = 4
    # max_rot_vel = 433.3

    # KF = 3.16e-10
    # KM = 7.94e-12
    # MAX_RPM = sqrt((thrust_to_weight*9.81) / (4*KF))

    # def initialize(self):
    #     super(MultirotorBase, self).initialize()
    #     self.base_link = RigidPrimView(
    #         prim_paths_expr=f"/World/envs/.*/{self.name}_*/body",
    #         name="base_link"
    #     )
    #     self.base_link.initialize()
    #     self.rotors = RigidPrimView(
    #         prim_paths_expr=f"/World/envs/.*/{self.name}_*/m[1-4]_prop",
    #         name="rotors"
    #     )
    #     self.rotors.initialize()
    #     self.tg = NoisyDampedMotor(
    #         {
    #             "max_thrust": 9.81 * self.mass * self.thrust_to_weight / self.num_rotors,
    #             "max_rot_vel": self.max_rot_vel,
    #             "rot_directions": self.rot_directions,
    #         },
    #         self.base_link,
    #         self.articulations,
    #         self.rotors,
    #         shape=[-1, self._count, self.num_rotors]
    #     )
    #     # local_translation = self.rotors.get_local_poses()[0]
    #     # local_translation[..., :-1] = 0
    #     # self.local_translation = -local_translation
    #     print(self.rotors.get_local_poses())
    
    # def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
    #     rpms = actions.clip(0, self.MAX_RPM)
    #     z_thrusts = rpms**2 * self.KF # (env, actor, 4)
    #     torques = rpms**2 * self.KM
    #     z_torques = (
    #         + torques[..., 0] 
    #         - torques[..., 1] 
    #         + torques[..., 2] 
    #         - torques[..., 3]
    #     )
    #     self.tg.thrusts[..., 2] = z_thrusts
    #     self.tg.torques[..., 2] = z_torques
        
    #     rpms = rpms * torch.pi / 30
    #     self.articulations.set_joint_velocities(rpms.reshape(-1, 4))
    #     self.rotors.apply_forces_and_torques_at_pos(
    #         self.tg.thrusts.reshape(-1, 3), None, is_global=False)
    #     self.base_link.apply_forces_and_torques_at_pos(
    #         None, self.tg.torques.reshape(-1, 3), is_global=False)
