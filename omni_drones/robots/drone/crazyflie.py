import torch
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.robots.robot import RobotBase
from omni_drones.actuators.damped_motor import NoisyDampedMotor

from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path

class Crazyflie(RobotBase):

    usd_path: str = get_assets_root_path() + "/Isaac/Robots/Crazyflie/cf2x.usd"
    mass = 0.028
    thrust_to_weight = 2.25

    def __init__(self, name: str="Crazyflie", cfg=None) -> None:
        super().__init__(name, cfg)
        self.action_spec = BoundedTensorSpec(-1, 1, (4,), device=self.device)
        self.state_spec = UnboundedContinuousTensorSpec(13, device=self.device)

    def initialize(self):
        super().initialize()
        self.rotors = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Crazyflie_*/m[1-4]_prop",
            name="rotors"
        )
        self.rotors.initialize()
        self.rotors.post_reset()
        self.tg = NoisyDampedMotor(
            {
                "thrust_max": 9.81 * self.mass * self.thrust_to_weight / 4, 
                "prop_max_rot": 433.3
            },
            self.articulations, 
            self.rotors,
            shape=[-1, self._count, 4]
        )

    def apply_action(self, actions: torch.Tensor):
        self.tg.apply_action(actions)
        return torch.square(actions).sum(-1)

    def _reset_idx(self, env_ids: torch.Tensor):
        self.tg.thrust_cmds_damp[env_ids] = 0
        self.tg.thrust_rot_damp[env_ids] = 0

