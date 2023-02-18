import torch
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path

from omni_drones.robots.robot import RobotBase
from omni_drones.actuators.damped_motor import NoisyDampedMotor
from omni_drones.robots.robot import RobotBase, ASSET_PATH

class Neo11(RobotBase):

    usd_path: str = ASSET_PATH + "/usd/neo11.usd"
    mass = 3.45
    thrust_to_weight = 2.25
    max_rot = 1047

    def __init__(self, name: str="Neo11", cfg=None) -> None:
        super().__init__(name, cfg)
        self.action_spec = BoundedTensorSpec(-1, 1, 6, device=self.device)
        self.state_spec = UnboundedContinuousTensorSpec(13 + 6, device=self.device)

    def initialize(self):
        super().initialize()
        self.rotors = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/{self.name}_*/rotor_[0-5]",
            name="rotors"
        )
        self.rotors.initialize()
        self.rotors.post_reset()
        self.tg = NoisyDampedMotor(
            {
                "thrust_max": 9.81 * self.mass * self.thrust_to_weight / 6, 
                "prop_max_rot": self.max_rot
            },
            self.articulations,
            self.rotors,
            shape=[-1, self._count, 6]
        )

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        self.tg.apply_action(actions)
        return torch.square(actions).sum(-1)
    
    def get_state(self):
        pos, rot = self.get_env_poses(False)
        vel = self.get_velocities(False)
        thr = self.tg.thrust_cmds_damp
        return torch.cat([pos, rot, vel, thr], dim=-1)
    
    def _reset_idx(self, env_ids: torch.Tensor):
        self.tg.thrust_cmds_damp[env_ids] = 0
        self.tg.thrust_rot_damp[env_ids] = 0

