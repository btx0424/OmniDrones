import torch
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.robots.robot import RobotBase
from omni_drones.actuators.thrust_generator import ThrustGenerator

from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path

class Quadcopter(RobotBase):

    usd_path: str = get_assets_root_path() + "/Isaac/Robots/Quadcopter/quadcopter.usd"

    def __init__(self, name: str="Quadcopter", cfg=None) -> None:
        super().__init__(name, cfg)
        self.action_spec = BoundedTensorSpec(-1, 1, (12,), device=self.device)
        self.state_spec = UnboundedContinuousTensorSpec(21, device=self.device)

    def initialize(self):
        super().initialize()
        self.rotors = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Quadcopter_*/rotor[0-3]",
            name="rotors",
            reset_xform_properties=False
        )
        self.rotors.initialize()
        self.rotors.post_reset()
        self.tg = ThrustGenerator(
            {
                "max_thrust": 2.0,
                "thrust_action_speed_scale" : 100
            },
            self.articulations, 
            self.rotors,
            shape=[-1, self._count, 4]
        )
        self.dof_position_targets = torch.zeros(
            (*self.shape, self.articulations.num_dof), device=self.device)
        
        dof_limits = self.articulations.get_dof_limits()
        self.dof_lower_limits = torch.as_tensor(dof_limits[0][:, 0], device=self.device)
        self.dof_upper_limits = torch.as_tensor(dof_limits[0][:, 1], device=self.device)
        
    def apply_action(self, actions: torch.Tensor):
        dof_action_speed_scale = 8 * torch.pi
        self.dof_position_targets += self.dt * dof_action_speed_scale * actions[..., 0:8]
        self.dof_position_targets.clamp_(self.dof_lower_limits, self.dof_upper_limits)
        self.articulations.set_joint_position_targets(self.dof_position_targets.flatten(0, -2))
        
        self.tg.apply_action(actions[..., 8:12])
        return torch.square(actions).sum(-1)

    def _reset_idx(self, env_ids: torch.Tensor):
        self.tg.thrust[env_ids] = 0
        self.dof_position_targets[env_ids] = 0

        dof_vel = self.get_joint_velocities()
        dof_vel[env_ids] = 0
        self.articulations.set_joint_velocities(dof_vel.flatten(0, -2))
        dof_pos = self.get_joint_positions()
        dof_pos[env_ids] = 0
        self.articulations.set_joint_positions(dof_pos.flatten(0, -2))

