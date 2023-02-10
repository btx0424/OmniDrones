import torch
from torchrl.data import BoundedTensorSpec

from omni_drones.robots.robot import RobotBase
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path

class Crazyflie(RobotBase):

    usd_path: str = get_assets_root_path() + "/Isaac/Robots/Crazyflie/cf2x.usd"
    action_spec = BoundedTensorSpec(-1, 1, (4,))

    def __init__(self, name: str="Crazyflie", cfg=None) -> None:
        super().__init__(name, cfg)

    def initialize(self):
        super().initialize()
        self.rotors = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/Crazyflie_*/m[1-4]_prop",
            name="rotors"
        )
        self.rotors.initialize()

    def apply_action(self, actions: torch.Tensor):
        thrusts = torch.zeros(self.count, 4, 3, device="cuda")
        thrusts[..., 2] = actions.to("cuda")
        self.rotors.apply_forces(thrusts)
