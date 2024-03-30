import torch

from omni_drones.robots.multirotor import Multirotor, Rotor

from .mdp_term import MDPTerm

class Randomization(MDPTerm):
    
    pass


class RandomizeRotorParams(Randomization):
    def __init__(
        self, 
        env: "IsaacEnv",
        asset_name: str = "drone",
        actuator_name: str = "rotor",
        kf_range: tuple[float, float] = (0.7, 1.3),
        km_range: tuple[float, float] = (0.7, 1.3),
        tau_range: tuple[float, float] = (0.4, 1.0),
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
        body_name: str = ".*",
        mass_range: tuple[float, float] = (0.5, 1.5),
    ):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]
        self.body_ids, self.body_names = self.asset.find_bodies(body_name)
        self.mass_range = mass_range

        masses = self.asset.root_physx_view.get_masses().clone()
        shape = (*self.asset.shape, len(self.body_ids))
        masses[:, self.body_ids] = uniform(shape, *mass_range, masses.device)
        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_masses(masses, indices)


def _get(t: torch.Tensor) -> torch.Tensor:
    return t.flatten(0, -2)[0].clone()


def uniform(size, low, high, device):
    return torch.rand(size, device=device) * (high - low) + low