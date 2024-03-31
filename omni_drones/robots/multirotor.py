import torch
import inspect
import functools

from omni.isaac.orbit.assets import Articulation, ArticulationData, ArticulationCfg
from omni.isaac.orbit.actuators import ActuatorBaseCfg, ActuatorBase
from omni.isaac.orbit.utils import configclass
import omni.isaac.orbit.utils.string as string_utils
from omni_drones.utils.torch import quaternion_to_euler, quat_rotate

from dataclasses import dataclass, MISSING, field
from typing import Sequence, Mapping


@dataclass
class MultirotorData(ArticulationData):

    rpy_w: torch.Tensor = None
    heading_w_vec: torch.Tensor = None

    throttle: Mapping[str, torch.Tensor] = field(default_factory=dict)
    drag_coef: torch.Tensor = None
    
    default_masses: torch.Tensor = None
    default_masses_total: torch.Tensor = None
    default_inertia: torch.Tensor = None

    applied_thrusts: Mapping[str, torch.Tensor] = field(default_factory=dict)
    applied_moments: Mapping[str, torch.Tensor] = field(default_factory=dict)


class Multirotor(Articulation):

    def __init__(self, cfg: ArticulationCfg):
        super().__init__(cfg)
        self._data = MultirotorData()
    
    @property
    def data(self):
        return self._data

    @property
    def env(self):
        return self._env

    @property
    def shape(self):
        if not hasattr(self, "_shape"):
            shape = torch.Size([self.env.num_envs, self.num_instances // self.env.num_envs])
            if shape[-1] == 1:
                shape = shape[:-1]
            self._shape = shape
        return self._shape

    def update(self, dt: float):
        super().update(dt)
        self._data.rpy_w[:] = quaternion_to_euler(self._data.root_quat_w)
        self._data.heading_w_vec[:] = quat_rotate(
            self._data.root_quat_w, 
            torch.tensor([1., 0., 0.], device=self.device)
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        for value in self._data.throttle.values():
            value[env_ids] = 0
    
    def write_data_to_sim(self):
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.
        """

        # apply actuator models first
        self._apply_actuator_model()

        # write external wrench
        if self.has_external_wrench:
            # apply external forces and torques
            self._body_physx_view.apply_forces_and_torques_at_position(
                force_data=self._external_force_body_view_b.view(-1, 3),
                torque_data=self._external_torque_body_view_b.view(-1, 3),
                position_data=None,
                indices=self._ALL_BODY_INDICES,
                is_global=False,
            )
    
    def _apply_actuator_model(self):
        # process actions per group
        for name, actuator in self.actuators.items():
            if isinstance(actuator, Rotor):
                thrusts, momentum = actuator.compute()
                self._data.throttle[name][:] = actuator.throttle
                self._data.applied_thrusts[name][:] = thrusts
                self._data.applied_moments[name][:] = momentum

                forces = torch.zeros_like(self._external_force_b)
                torques = torch.zeros_like(self._external_torque_b)
                
                thrusts = thrusts.reshape(self.num_instances, len(actuator.body_ids))
                momentum = momentum.reshape(self.num_instances, len(actuator.body_ids))
                
                forces[..., actuator.body_ids, 2] += thrusts
                # torques[..., actuator.body_ids, 2] = momentum
                # mannually aggregate the torques along the z-axis
                torques[..., self.base_id, 2] += momentum.sum(dim=-1, keepdim=True)
                
                drag = (
                    self._data.drag_coef.unsqueeze(-1)
                    * -self._data.body_lin_vel_w 
                    * self._data.default_masses_total.unsqueeze(-1)
                )
                forces.add_(drag)

                self.set_external_force_and_torque(
                    forces=forces,
                    torques=torques,
                )
            else:
                pass
    
    def _process_actuators_cfg(self):
        for actuator_name, actuator_cfg in list(self.cfg.actuators.items()):
            if isinstance(actuator_cfg, RotorCfg):
                actuator_class = actuator_cfg.class_type
                actuator: Rotor = actuator_class(cfg=actuator_cfg, articulation=self)
                self.actuators[actuator_name] = actuator

                # create data for the actuator
                self._data.throttle[actuator_name] = torch.zeros(actuator.shape, device=self.device)
                self._data.applied_thrusts[actuator_name] = torch.zeros(actuator.shape, device=self.device)
                self._data.applied_moments[actuator_name] = torch.zeros(actuator.shape, device=self.device)
                
                del self.cfg.actuators[actuator_name]
            
        super()._process_actuators_cfg()

    def _initialize_impl(self): 
        super()._initialize_impl()
        self.base_id, self.base_name = self.find_bodies("base_link")
        self._data.rpy_w = torch.zeros(self.shape + (3,), device=self.device).flatten(0, -2)
        self._data.heading_w_vec = torch.zeros(self.shape + (3,), device=self.device).flatten(0, -2)
        
        self._data.default_masses = self.root_physx_view.get_masses().clone()
        self._data.default_masses_total = self._data.default_masses.sum(dim=-1, keepdim=True).to(self.device)
        self._data.default_inertia = self.root_physx_view.get_inertias()[:, self.base_id[0], [0, 4, 8]].clone()
        self._data.drag_coef = torch.zeros(*self.shape, self.num_bodies, device=self.device)

    def resolve_ids(self, env_ids: torch.Tensor):
        return self._ALL_INDICES.reshape(self.shape)[env_ids].flatten()


class Rotor(ActuatorBase):

    mapping = torch.square

    def __init__(
        self, 
        cfg: "RotorCfg",
        articulation: Multirotor,
    ):
        super().__init__(cfg, [], [], articulation.env.num_envs, device=articulation.device)
        self.asset = articulation

        self.cfg: RotorCfg

        self.body_ids, self.body_names = self.asset.find_bodies(self.cfg.body_names_expr)
        self._shape = (*self.asset.shape, len(self.body_ids))

        with torch.device(self.device):
            
            def resolve(value_mapping):
                if not isinstance(value_mapping, Mapping):
                    value_mapping = {name: value_mapping for name in self.body_names}
                indices, names, values = string_utils.resolve_matching_names_values(value_mapping, self.body_names)
                tensor = torch.as_tensor(values).expand(self.shape).clone()
                assert tensor.shape == self.shape
                return tensor
            
            self.throttle = torch.zeros(self.shape, device=self.device)
            self.throttle_target = torch.zeros(self.shape, device=self.device)

            self.max_rotor_speed = resolve(self.cfg.max_rotor_speed)
            self.kf_normalized = resolve(self.cfg.kf) * self.mapping(self.max_rotor_speed)
            self.km_normalized = resolve(self.cfg.km) * self.mapping(self.max_rotor_speed)
            self.rotor_direction = resolve(self.cfg.rotor_direction)
            self.tau_up = resolve(self.cfg.tau_up)
            self.tau_down = resolve(self.cfg.tau_down)
    
    @property
    def device(self):
        return self.asset.device
    
    @property
    def num_rotors(self):
        return len(self.body_ids)
    
    @property
    def shape(self):
        return self._shape
    
    def reset(self, env_ids: Sequence[int]):
        self.throttle[env_ids] = 0
        self.throttle_target[env_ids] = 0

    def compute(self):
        tau = torch.where(self.throttle_target > self.throttle, self.tau_up, self.tau_down)
        self.throttle.add_(tau * (self.throttle_target - self.throttle)).clamp_(0, 1)

        mapped = self.throttle
        thrusts = self.kf_normalized * mapped
        moments = self.km_normalized * mapped * -self.rotor_direction

        return thrusts, moments

    def _parse_joint_parameter(self, cfg_value: float | dict[str, float] | None, default_value: float | torch.Tensor | None) -> torch.Tensor:
        return None


@configclass
class RotorCfg(ActuatorBaseCfg):

    class_type: type[ActuatorBase] = Rotor
    body_names_expr: list[str] = MISSING
    
    max_rotor_speed: float | Mapping[str, float] = MISSING
    """
    The maximum rotor speed.
    """

    rotor_direction: list[int] | Mapping[str, list[int]] = MISSING
    """
    The rotor directions.
    """

    kf: float | Mapping[str, float] = MISSING
    """
    The thrust coefficient.
    """

    km: float | Mapping[str, float] = MISSING
    """
    The moment coefficient.
    """

    tau_up: float | Mapping[str, float] = 0.5
    """
    The throttle up time constant.
    """

    tau_down: float | Mapping[str, float] = 0.5
    """
    The throttle down time constant.
    """



