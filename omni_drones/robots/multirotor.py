import torch
import inspect
import functools

from omni.isaac.lab.assets import Articulation, ArticulationData, ArticulationCfg
from omni.isaac.lab.actuators import ActuatorBaseCfg, ActuatorBase
from omni.isaac.lab.utils import configclass
from omni.isaac.core.utils.types import ArticulationActions
import omni.isaac.lab.utils.string as string_utils

from omni_drones.utils.torch import quaternion_to_euler, quat_rotate, quat_rotate_inverse

from dataclasses import dataclass, MISSING, field
from typing import Sequence, Mapping


@dataclass
class MultirotorData:

    articulation: ArticulationData

    rpy_w: torch.Tensor = None
    heading_w_vec: torch.Tensor = None

    throttle: Mapping[str, torch.Tensor] = field(default_factory=dict)
    drag_coef: torch.Tensor = None

    applied_thrusts: Mapping[str, torch.Tensor] = field(default_factory=dict)
    applied_moments: Mapping[str, torch.Tensor] = field(default_factory=dict)
    applied_drag_b: torch.Tensor = None


class _View:
    def __init__(self, data, shape):
        self.data = data
        self.shape = shape

    def __getattr__(self, key):
        value = getattr(self.data, key)
        if isinstance(value, torch.Tensor):
            return value.view(self.shape + value.shape[1:])
        else:
            return value

class Multirotor(Articulation):

    @functools.cached_property
    def data(self) -> ArticulationData:
        data = _View(self._data, self.shape)
        return data
    
    @property
    def multirotor_data(self):
        if not hasattr(self, "_multirotor_data"):
            self._multirotor_data = MultirotorData(self)
        return self._multirotor_data

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

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        for value in self.multirotor_data.throttle.values():
            value[env_ids] = 0
    
    def write_data_to_sim(self):
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.
        """

        forces_b = torch.zeros_like(self._external_force_b)
        torques_b = torch.zeros_like(self._external_torque_b)

        for name, actuator in self.actuators.items():
            if isinstance(actuator, Rotor):
                thrusts, momentum = actuator.rotor_compute()
                self.multirotor_data.throttle[name][:] = actuator.throttle
                self.multirotor_data.applied_thrusts[name][:] = thrusts
                self.multirotor_data.applied_moments[name][:] = momentum
                
                thrusts = thrusts.reshape(self.num_instances, len(actuator.body_ids))
                momentum = momentum.reshape(self.num_instances, len(actuator.body_ids))
                
                forces_b[..., actuator.body_ids, 2] += thrusts
                # torques[..., actuator.body_ids, 2] = momentum
                # mannually aggregate the torques along the z-axis
                torques_b[..., self.base_id, 2] += momentum.sum(dim=-1, keepdim=True)
        
        drag_w = (
            self.multirotor_data.drag_coef.unsqueeze(-1)
            * -self.data.body_lin_vel_w
        )
        self.multirotor_data.applied_drag_b[:] = quat_rotate_inverse(
            self.data.body_quat_w, # [*, body, 4]
            drag_w # [*, body, 3]
        )

        if len(self.shape) > 1:
            forces_b += self.multirotor_data.applied_drag_b.flatten(0, 1)
        else:
            forces_b += self.multirotor_data.applied_drag_b

        self.set_external_force_and_torque(forces=forces_b, torques=torques_b)
        super().write_data_to_sim()
    
    def _process_actuators_cfg(self):
        print("[INFO] Processing Isaac Articulation Actuators.")
        # collect and remove RotorCfg from `self.cfg.actuators`
        # since the super class only processes IsaacSim Articulation actuators
        actuators = {}
        for actoror_name, actoror_cfg in list(self.cfg.actuators.items()):
            if isinstance(actoror_cfg, RotorCfg):
                actuators[actoror_name] = actoror_cfg
                del self.cfg.actuators[actoror_name]
        super()._process_actuators_cfg()
        print("[INFO]: Processing OmniDrones Actuators.")
        for actuator_name, actuator_cfg in actuators.items():
            if isinstance(actuator_cfg, RotorCfg):
                actuator_class = actuator_cfg.class_type
                actuator: Rotor = actuator_class(cfg=actuator_cfg, articulation=self)
                self.actuators[actuator_name] = actuator

                # create data for the actuator
                self.multirotor_data.throttle[actuator_name] = torch.zeros(actuator.shape, device=self.device)
                self.multirotor_data.applied_thrusts[actuator_name] = torch.zeros(actuator.shape, device=self.device)
                self.multirotor_data.applied_moments[actuator_name] = torch.zeros(actuator.shape, device=self.device)
                
    def _initialize_impl(self): 
        super()._initialize_impl()
        self.base_id, self.base_name = self.find_bodies("base_link")
        # self._data.rpy_w = torch.zeros(self.shape + (3,), device=self.device).flatten(0, -2)
        # self._data.heading_w_vec = torch.zeros(self.shape + (3,), device=self.device).flatten(0, -2)

        self.multirotor_data.drag_coef = torch.zeros(*self.shape, self.num_bodies, device=self.device)
        self.multirotor_data.applied_drag_b = torch.zeros(*self.shape, self.num_bodies, 3, device=self.device)

    def _apply_actuator_model(self):
        """Processes joint commands for the articulation by forwarding them to the actuators.

        The actions are first processed using actuator models. Depending on the robot configuration,
        the actuator models compute the joint level simulation commands and sets them into the PhysX buffers.
        """
        # process actions per group
        for actuator in self.actuators.values():
            if isinstance(actuator, Rotor): continue
            # prepare input for actuator model based on cached data
            # TODO : A tensor dict would be nice to do the indexing of all tensors together
            control_action = ArticulationActions(
                joint_positions=self._data.joint_pos_target[:, actuator.joint_indices],
                joint_velocities=self._data.joint_vel_target[:, actuator.joint_indices],
                joint_efforts=self._data.joint_effort_target[:, actuator.joint_indices],
                joint_indices=actuator.joint_indices,
            )
            # compute joint command from the actuator model
            control_action = actuator.compute(
                control_action,
                joint_pos=self._data.joint_pos[:, actuator.joint_indices],
                joint_vel=self._data.joint_vel[:, actuator.joint_indices],
            )
            # update targets (these are set into the simulation)
            if control_action.joint_positions is not None:
                self._joint_pos_target_sim[:, actuator.joint_indices] = control_action.joint_positions
            if control_action.joint_velocities is not None:
                self._joint_vel_target_sim[:, actuator.joint_indices] = control_action.joint_velocities
            if control_action.joint_efforts is not None:
                self._joint_effort_target_sim[:, actuator.joint_indices] = control_action.joint_efforts
            # update state of the actuator model
            # -- torques
            self._data.computed_torque[:, actuator.joint_indices] = actuator.computed_effort
            self._data.applied_torque[:, actuator.joint_indices] = actuator.applied_effort
            # -- actuator data
            self._data.soft_joint_vel_limits[:, actuator.joint_indices] = actuator.velocity_limit
            # TODO: find a cleaner way to handle gear ratio. Only needed for variable gear ratio actuators.
            if hasattr(actuator, "gear_ratio"):
                self._data.gear_ratio[:, actuator.joint_indices] = actuator.gear_ratio

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

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> ArticulationActions:
        return control_action

    def rotor_compute(self):
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



