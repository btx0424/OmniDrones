import torch
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni.isaac.core.prims import RigidPrimView

from omni_drones.robots.robot import RobotBase
from omni_drones.actuators.damped_motor import NoisyDampedMotor

class MultirotorBase(RobotBase):

    mass: float
    thrust_to_weight: float
    num_rotors : int
    max_rot_vel: float
    rot_directions = None

    def __init__(self, name: str=None, cfg=None) -> None:
        super().__init__(name, cfg)
        self.action_spec = BoundedTensorSpec(-1, 1, self.num_rotors, device=self.device)
        self.state_spec = UnboundedContinuousTensorSpec(13 + self.num_rotors, device=self.device)
    
    def initialize(self):
        super().initialize()
        self.base_link = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/{self.name}_*/base_link",
            name="base_link"
        )
        self.base_link.initialize()
        self.rotors = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/{self.name}_*/rotor_[0-{self.num_rotors-1}]",
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