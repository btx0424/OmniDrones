import torch

from omni_drones.robots.multirotor import Multirotor, Rotor
import omni.isaac.orbit.utils.string as string_utils

from typing import Dict, Tuple
from .mdp_term import MDPTerm

RangeType = Tuple[float, float]

class Randomization(MDPTerm):
    
    pass


class RandomizeRotorParams(Randomization):
    def __init__(
        self, 
        env: "IsaacEnv",
        asset_name: str = "drone",
        actuator_name: str = "rotor",
        kf_range: RangeType = (0.7, 1.3),
        km_range: RangeType = (0.7, 1.3),
        tau_range: RangeType = (0.4, 1.0),
    ):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]
        self.actuator: Rotor = self.asset.actuators[actuator_name]

        with torch.device(self.device):
            self.kf_range = torch.as_tensor(kf_range)
            self.km_range = torch.as_tensor(km_range)
            self.tau_range = torch.as_tensor(tau_range)

        self.default_kf = _get(self.actuator.kf_normalized)
        self.default_km = _get(self.actuator.km_normalized)
    
    def reset(self, env_ids: torch.Tensor):
        shape = (len(env_ids), *self.actuator.shape[1:])
        
        tau = uniform(shape, *self.tau_range, self.device)
        kf = uniform(shape, *self.kf_range, self.device)
        km = uniform(shape, *self.km_range, self.device)
        
        self.actuator.tau_up[env_ids] = tau
        self.actuator.tau_down[env_ids] = tau

        self.actuator.kf_normalized[env_ids] = kf * self.default_kf
        self.actuator.km_normalized[env_ids] = km * self.default_km


class RandomizeBodyMass(Randomization):

    def __init__(
        self, 
        env: "IsaacEnv",
        asset_name: str = "drone",
        mass_range: Dict[str, RangeType] | RangeType = (0.5, 1.5),
    ):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]
        self.mass_range = mass_range

        if isinstance(mass_range, tuple):
            mass_range = {".*": mass_range}
        
        low = {k: v[0] for k, v in mass_range.items()}
        high = {k: v[1] for k, v in mass_range.items()}
        
        self.body_ids, _, self.low = string_utils.resolve_matching_names_values(low, self.asset.body_names)
        self.body_ids, _, self.high = string_utils.resolve_matching_names_values(high, self.asset.body_names)
        
        self.low = torch.as_tensor(self.low)
        self.high = torch.as_tensor(self.high)

        masses = self.asset.root_physx_view.get_masses().clone()
        shape = (*self.asset.shape, len(self.body_ids))
        masses[:, self.body_ids] = uniform(shape, self.low, self.high, masses.device)
        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_masses(masses, indices)


class RandomizeBodyInertia(Randomization):
    
    def __init__(
        self, 
        env: "IsaacEnv",
        asset_name: str = "drone",
        inertia_range: Dict[str, RangeType] | RangeType = None,
        inertia_ratio_range: Dict[str, RangeType] | RangeType = None,
    ):
        super().__init__(env)
        if (inertia_range is None) == (inertia_ratio_range is None):
            raise ValueError("Exactly one of inertia_range and inertia_ratio_range must be provided")
        
        self.asset: Multirotor = self.env.scene[asset_name]
        self.inertia_range = inertia_range

        if isinstance(inertia_range, tuple):
            inertia_range = {".*": inertia_range}
        
        inertias = self.asset.root_physx_view.get_inertias().clone()

        if inertia_range is not None:
            low = {k: v[0] for k, v in inertia_range.items()}
            high = {k: v[1] for k, v in inertia_range.items()}

            self.body_ids, _, self.low = string_utils.resolve_matching_names_values(low, self.asset.body_names)
            self.body_ids, _, self.high = string_utils.resolve_matching_names_values(high, self.asset.body_names)
            self.low = torch.as_tensor(self.low)
            self.high = torch.as_tensor(self.high)
        else:
            low = {k: v[0] for k, v in inertia_ratio_range.items()}
            high = {k: v[1] for k, v in inertia_ratio_range.items()}
        
            self.body_ids, _, self.low = string_utils.resolve_matching_names_values(low, self.asset.body_names)
            self.body_ids, _, self.high = string_utils.resolve_matching_names_values(high, self.asset.body_names)
            
            index = (
                torch.arange(self.asset.num_instances).reshape(-1, 1, 1),
                torch.as_tensor(self.body_ids).reshape(-1, 1), 
                torch.as_tensor([0, 4, 8])
            )
            self.low = torch.as_tensor(self.low) * inertias[index]
            self.high = torch.as_tensor(self.high) * inertias[index]

        shape = (*self.asset.shape, len(self.body_ids), 3)
        inertias[index] = uniform(shape, self.low, self.high, inertias.device)
        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_inertias(inertias, indices)

class RandomizeDragCoef(Randomization):

    def __init__(
        self, 
        env: "IsaacEnv",
        asset_name: str = "drone",
        drag_coef_range: Dict[str, RangeType] | RangeType = (0.5, 1.5),
    ):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]
        self.drag_coef_range = drag_coef_range

        if isinstance(drag_coef_range, tuple):
            drag_coef_range = {".*": drag_coef_range}
        
        low = {k: v[0] for k, v in drag_coef_range.items()}
        high = {k: v[1] for k, v in drag_coef_range.items()}
        
        self.body_ids, _, self.low = string_utils.resolve_matching_names_values(low, self.asset.body_names)
        self.body_ids, _, self.high = string_utils.resolve_matching_names_values(high, self.asset.body_names)
        
        self.body_ids = torch.as_tensor(self.body_ids, device=self.asset.device)
        self.low = torch.as_tensor(self.low, device=self.asset.device)
        self.high = torch.as_tensor(self.high, device=self.asset.device)

    def reset(self, env_ids: torch.Tensor):
        shape = (len(env_ids), len(self.body_ids))
        index = (env_ids.unsqueeze(1), self.body_ids)
        self.asset._data.drag_coef[index] = uniform(shape, self.low, self.high, self.asset.device)


def _get(t: torch.Tensor) -> torch.Tensor:
    return t.flatten(0, -2)[0].clone()


def uniform(size, low, high, device):
    return torch.rand(size, device=device) * (high - low) + low