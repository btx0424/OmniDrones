from omni_drones.robots.robot import RobotBase, ASSET_PATH
from omni_drones.robots.drone import MultirotorBase
import torch
from functorch import vmap
from torchrl.data import BoundedTensorSpec

class Omav(MultirotorBase):

    usd_path: str = ASSET_PATH + "/usd/omav.usd"

    def __init__(self, name: str="Omav", cfg=None) -> None:
        super().__init__(name, cfg)
        self.action_spec = BoundedTensorSpec(-1, 1, 12 + 6, device=self.device)

    def initialize(self, prim_paths_expr: str = None):
        super().initialize(prim_paths_expr)

        print(self._view.dof_names)
        print(self._view._dof_types)
        print(self._view._dof_indices)

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        # actions dim: 12+6=18, each range in [-1,1]
        
        rotor_cmds, tilt_cmds = (
            actions.expand(*self.shape, self.num_rotors)
            .split([12, 6], dim=-1)
        )

        thrusts, moments = vmap(vmap(self.rotors))(rotor_cmds, self.rotor_params_and_states)
        self.forces[..., 2] = thrusts
        self.torques[..., 2] = moments.sum(-1)
        self.rotors_view.apply_forces(self.forces.reshape(-1, 3), is_global=False)

        self.base_link.apply_forces_and_torques_at_pos(
            None, self.torques.reshape(-1, 3), is_global=False)
        return self.throttle.sum(-1)