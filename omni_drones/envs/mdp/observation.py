from omni_drones.envs.isaac_env import IsaacEnv
import torch
from omni_drones.robots.multirotor import Multirotor, Rotor

from .mdp_term import MDPTerm


class ObservationFunc(MDPTerm):
    pass


class DynamicState(ObservationFunc):

    def __init__(self, env: "IsaacEnv", asset_name: str="drone"):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]

    def compute(self) -> torch.Tensor:
        state = torch.cat([
            self.asset.data.root_pos_w[..., [2]],
            self.asset.data.root_quat_w,
            self.asset.data.root_vel_b,
            self.asset.data.projected_gravity_b,
            *self.asset.data.throttle.values()
        ], dim=-1)
        return state.reshape(*self.asset.shape, -1)


class BodyDynamicState(ObservationFunc):
    
    def __init__(
        self, 
        env: "IsaacEnv", 
        asset_name: str="drone",
        body_name: str="payload"
    ):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]
        self.body_id = self.asset.find_bodies(body_name)[0]
    
    def compute(self) -> torch.Tensor:
        body_pos = (
            self.asset.data.body_pos_w[..., self.body_id, :]
            - self.asset.data.root_pos_w.unsqueeze(-2)
        )
        state = torch.cat([
            body_pos,
            self.asset.data.body_quat_w[..., self.body_id, :],
            self.asset.data.body_vel_w[..., self.body_id, :],
        ], dim=-1)
        return state.reshape(*self.asset.shape, -1)


class RotorParam(ObservationFunc):

    def __init__(
        self, 
        env: "IsaacEnv", 
        asset_name: str="drone",
        actuator_name: str="rotor"
    ):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]
        self.actuator: Rotor = self.asset.actuators[actuator_name]

        self.default_kf = _get(self.actuator.kf_normalized)
        self.default_km = _get(self.actuator.km_normalized)
    
    def compute(self) -> torch.Tensor:
        params = torch.cat([
            self.actuator.kf_normalized / self.default_kf,
            self.actuator.km_normalized / self.default_km,
            self.actuator.tau_up,
            self.actuator.tau_down,
        ], dim=-1)
        return params.reshape(*self.asset.shape, -1)


class BodyMass(ObservationFunc):
    def __init__(
        self, 
        env: "IsaacEnv",
        asset_name: str="drone",
        body_name: str="payload"
    ):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]
        self.body_ids, self.body_names = self.asset.find_bodies(body_name)

        masses = self.asset.data.default_masses.to(self.device)
        self.body_masses = (
            (masses[..., self.body_ids] / self.asset.data.default_masses_total)
            .reshape(*self.asset.shape, len(self.body_ids))
        )

    def compute(self) -> torch.Tensor:
        return self.body_masses


class DragCoef(ObservationFunc):
        
        def __init__(
            self, 
            env: "IsaacEnv", 
            asset_name: str="drone",
            body_name: str="payload"
        ):
            super().__init__(env)
            self.asset: Multirotor = self.env.scene[asset_name]
            self.body_ids, self.body_names = self.asset.find_bodies(body_name)
            
        def compute(self) -> torch.Tensor:
            return (
                self.asset._data.drag_coef[:, self.body_ids]
                .reshape(*self.asset.shape, -1)
            )


class AppliedActuation(ObservationFunc):
    
        def __init__(
            self, 
            env: "IsaacEnv", 
            asset_name: str="drone",
            actuator_name: str="rotor"
        ):
            super().__init__(env)
            self.asset: Multirotor = self.env.scene[asset_name]
            if self.asset.actuators.get(actuator_name) is None:
                raise ValueError(f"Actuator {actuator_name} not found in asset {asset_name}")
            
            self.default_masses_total = self.asset.data.default_masses_total.to(self.device)
            self.default_inertia = self.asset.data.default_inertia.to(self.device)

            self.applied_thrusts = self.asset.data.applied_thrusts[actuator_name]
            self.applied_moments = self.asset.data.applied_moments[actuator_name]

        def compute(self) -> torch.Tensor:
            actuation = torch.cat([
                self.applied_thrusts / self.default_masses_total,
                self.applied_moments / self.default_inertia[..., [2]],
            ], dim=-1)
            return actuation.reshape(*self.asset.shape, -1)


def _get(t: torch.Tensor) -> torch.Tensor:
    return t.flatten(0, -2)[0].clone()