import logging
from typing import Type, Dict

import omni.isaac.core.utils.torch as torch_utils
import torch
import torch.distributions as D
import yaml
from functorch import vmap
from tensordict.nn import make_functional
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDict

from omni_drones.views import RigidPrimView
from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.controllers import LeePositionController

from omni_drones.robots import RobotBase, RobotCfg
from omni_drones.utils.torch import normalize, off_diag

from dataclasses import dataclass
from collections import defaultdict

@dataclass
class MultirotorCfg(RobotCfg):
    force_sensor: bool = False

class MultirotorBase(RobotBase):

    param_path: str
    DEFAULT_CONTROLLER: Type = LeePositionController
    cfg_cls = MultirotorCfg

    def __init__(
        self, 
        name: str = None, 
        cfg: MultirotorCfg=None, 
        is_articulation: bool = True
    ) -> None:
        super().__init__(name, cfg, is_articulation)

        with open(self.param_path, "r") as f:
            logging.info(f"Reading {self.name}'s params from {self.param_path}.")
            self.params = yaml.safe_load(f)
        self.num_rotors = self.params["rotor_configuration"]["num_rotors"]

        self.action_spec = BoundedTensorSpec(-1, 1, self.num_rotors, device=self.device)
        if self.cfg.force_sensor:
            self.use_force_sensor = True
            state_dim = 19 + self.num_rotors + 6
        else:
            self.use_force_sensor = False
            state_dim = 19 + self.num_rotors
        self.state_spec = UnboundedContinuousTensorSpec(state_dim, device=self.device)
        self.randomization = defaultdict(dict)

    def initialize(self, prim_paths_expr: str = None):
        if self.is_articulation:
            super().initialize(prim_paths_expr=prim_paths_expr)
            self.base_link = RigidPrimView(
                prim_paths_expr=f"{self.prim_paths_expr}/base_link",
                name="base_link",
                # track_contact_forces=True,
                shape=self.shape,
            )
            self.base_link.initialize()
            print(self._view.dof_names)
            print(self._view._dof_indices)
        else:
            super().initialize(prim_paths_expr=f"{prim_paths_expr}/base_link")
            self.base_link = self._view
            self.prim_paths_expr = prim_paths_expr

        self.rotors_view = RigidPrimView(
            # prim_paths_expr=f"{self.prim_paths_expr}/rotor_[0-{self.num_rotors-1}]",
            prim_paths_expr=f"{self.prim_paths_expr}/rotor_*",
            name="rotors",
            shape=(*self.shape, self.num_rotors)
        )
        self.rotors_view.initialize()

        self.rotors = RotorGroup(self.params["rotor_configuration"], dt=self.dt).to(
            self.device
        )
        rotor_params = make_functional(self.rotors)
        self.KF_0 = rotor_params["KF"].clone()
        self.KM_0 = rotor_params["KM"].clone()
        self.rotor_params = rotor_params.expand(self.shape).clone()

        self.KF = self.rotor_params["KF"]
        self.KM = self.rotor_params["KM"]
        self.throttle = self.rotor_params["throttle"]
        self.directions = self.rotor_params["directions"]

        self.thrusts = torch.zeros(*self.shape, self.num_rotors, 3, device=self.device)
        self.torques = torch.zeros(*self.shape, 3, device=self.device)
        self.forces = torch.zeros(*self.shape, 3, device=self.device)

        self.pos, self.rot = self.get_world_poses(True)
        self.throttle_difference = torch.zeros(self.throttle.shape[:-1], device=self.device)
        self.heading = torch.zeros(*self.shape, 3, device=self.device)
        self.up = torch.zeros(*self.shape, 3, device=self.device)
        self.vel = torch.zeros(*self.shape, 6, device=self.device)
        self.acc = torch.zeros(*self.shape, 6, device=self.device)
        self.jerk = torch.zeros(*self.shape, 6, device=self.device)
        self.alpha = 0.9

        self.rotor_pos_0 = (
            self.rotors_view.get_world_poses()[0][0] 
            - self.pos[0].unsqueeze(1)
        )
        self.rotor_pos_offset = torch.zeros(*self.shape, self.num_rotors, 3, device=self.device)

        self.masses = self.base_link.get_masses().clone()
        self.mass_0 = self.masses[0].clone()
        self.inertia_0 = self.base_link.get_inertias().reshape(-1, 3, 3)[0].diagonal().clone()
        self.drag_coef = torch.zeros(*self.shape, device=self.device) * self.params["drag_coef"]
        self.info_spec = CompositeSpec({
            "mass": UnboundedContinuousTensorSpec(1),
            "inertia": UnboundedContinuousTensorSpec(3),
            # "com": UnboundedContinuousTensorSpec(3),
            "KF": UnboundedContinuousTensorSpec(self.num_rotors),
            "KM": UnboundedContinuousTensorSpec(self.num_rotors),
            "drag_coef": UnboundedContinuousTensorSpec(1),
            "rotor_pos_offset": UnboundedContinuousTensorSpec(1),
        }).expand(self.shape).to(self.device)
        self.info = TensorDict({
            "mass": (self.masses / self.mass_0).unsqueeze(-1),
            "inertia": self.base_link.get_inertias().reshape(*self.shape, 3, 3).diagonal(0, -2, -1) / self.inertia_0,
            # "com": self.base_link.get_coms()[0] / self.params["l"],
            "KF": self.KF / self.KF_0,
            "KM": self.KM / self.KM_0,
            "drag_coef": self.drag_coef.unsqueeze(-1),
            "rotor_pos_offset": torch.zeros(*self.shape, 1, device=self.device),
        }, self.shape)

    def setup_randomization(self, cfg):
        if not self.initialized:
            raise RuntimeError
        
        from omegaconf import OmegaConf
        logging.info(f"Setup randomization:\n {OmegaConf.to_yaml(cfg)}")

        for phase in ("train", "eval"):
            if phase not in cfg: continue
            mass_scale = cfg[phase].get("mass_scale", None)
            if mass_scale is not None:
                low = self.params["mass"] * mass_scale[0]
                high = self.params["mass"] * mass_scale[1]
                self.randomization[phase]["mass"] = D.Uniform(
                    torch.tensor(low, device=self.device),
                    torch.tensor(high, device=self.device)
                )
            inertia_scale = cfg[phase].get("inertia_scale", None)
            if inertia_scale is not None:
                low = torch.as_tensor(inertia_scale[0], device=self.device)
                high = torch.as_tensor(inertia_scale[1], device=self.device)
                self.randomization[phase]["inertia"] = D.Uniform(
                    self.inertia_0 * low,
                    self.inertia_0 * high
                )
            com_scale= cfg[phase].get("com_scale", None)
            if com_scale is not None:
                self.randomization[phase]["com"] = D.Uniform(
                    -torch.as_tensor(com_scale, device=self.device) * self.params["l"],
                    torch.as_tensor(com_scale, device=self.device) * self.params["l"]
                )
            KM_scale = cfg[phase].get("KM_scale", None)
            if KM_scale is not None:
                self.randomization[phase]["KM"] = D.Uniform(
                    self.KM_0 * KM_scale[0],
                    self.KM_0 * KM_scale[1] 
                )
            KF_scale = cfg[phase].get("KF_scale", None)
            if KF_scale is not None:
                self.randomization[phase]["KF"] = D.Uniform(
                    self.KF_0 * KF_scale[0],
                    self.KF_0 * KF_scale[1] 
                )
            drag_coef_scale = cfg[phase].get("drag_coef_scale", None)
            if drag_coef_scale is not None:
                low = self.params["drag_coef"] * drag_coef_scale[0]
                high = self.params["drag_coef"] * drag_coef_scale[1]
                self.randomization[phase]["drag_coef"] = D.Uniform(
                    torch.tensor(low, device=self.device),
                    torch.tensor(high, device=self.device)
                )
            rotor_pos_offset_scale = cfg[phase].get("rotor_pos_offset_scale")
            if rotor_pos_offset_scale is not None:
                self.randomization[phase]["rotor_pos_offset"] = D.Normal(
                    torch.tensor([0], device=self.device), 
                    torch.tensor(rotor_pos_offset_scale, device=self.device)
                )
            if not len(self.randomization[phase]) == len(cfg[phase]):
                raise ValueError(f"Unknown randomization in {cfg[phase]}.")

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds = actions.expand(*self.shape, self.num_rotors)
        last_throttle = self.throttle.clone()
        thrusts, moments = vmap(vmap(self.rotors, randomness="different"), randomness="same")(
            rotor_cmds, self.rotor_params
        )

        rotor_pos, rotor_rot = self.rotors_view.get_world_poses()
        torque_axis = torch_utils.quat_axis(
            rotor_rot.flatten(end_dim=-2), axis=2
        ).unflatten(0, (*self.shape, self.num_rotors))

        self.thrusts[..., 2] = thrusts
        self.torques[:] = (moments.unsqueeze(-1) * torque_axis).sum(-2)
        # self.articulations.set_joint_velocities(
        #     (self.throttle * self.directions * self.MAX_ROT_VEL).reshape(-1, self.num_rotors)
        # )
        self.forces.zero_()
        # TODO: global downwash
        if self.n > 1:
            self.forces[:] += vmap(self.downwash)(
                self.pos,
                self.pos,
                vmap(torch_utils.quat_rotate)(self.rot, self.thrusts.sum(-2)),
                kz=0.3
            ).sum(-2)
        self.forces[:] += (self.drag_coef * self.masses).unsqueeze(-1) * self.vel[..., :3]

        self.rotors_view.apply_forces_and_torques_at_pos(
            self.thrusts.reshape(-1, 3), 
            positions=self.rotor_pos_offset,
            is_global=False
        )
        self.base_link.apply_forces_and_torques_at_pos(
            vmap(torch_utils.quat_rotate)(self.rot, self.forces).reshape(-1, 3), 
            self.torques.reshape(-1, 3), 
            is_global=True
        )
        self.throttle_difference[:] = torch.norm(self.throttle - last_throttle, dim=-1)
        return self.throttle.sum(-1)

    def get_state(self, env=True, check_nan: bool=False):
        self.pos[:], self.rot[:] = self.get_world_poses(True)
        if env:
            self.pos[:] = self.pos[:] - RobotBase._envs_positions
        vel = self.get_velocities(True)
        acc = self.acc.lerp((vel - self.vel) / self.dt, self.alpha)
        jerk = self.jerk.lerp((acc - self.acc) / self.dt, self.alpha)
        self.jerk[:] = jerk
        self.acc[:] = acc
        self.vel[:] = vel
        self.heading[:] = vmap(torch_utils.quat_axis)(self.rot, axis=0)
        self.up[:] = vmap(torch_utils.quat_axis)(self.rot, axis=2)
        state = [self.pos, self.rot, self.vel, self.heading, self.up, self.throttle * 2 - 1]
        if self.use_force_sensor:
            self.force_sensor_readings = self.get_force_sensor_forces() 
            state.append(self.force_sensor_readings.flatten(-2)/ self.masses.unsqueeze(-1))
        state = torch.cat(state, dim=-1)
        if check_nan:
            assert not torch.isnan(state).any()
        return state

    def get_info(self):
        info = torch.cat(list(self.info.values()), dim=-1)
        return info

    def _reset_idx(self, env_ids: torch.Tensor, train: bool=True):
        if env_ids is None:
            env_ids = torch.arange(self.shape[0], device=self.device)
        self.thrusts[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.vel[env_ids] = 0.
        self.acc[env_ids] = 0.
        self.jerk[env_ids] = 0.
        if train and "train" in self.randomization:
            self._randomize(env_ids, self.randomization["train"])
        elif "eval" in self.randomization:
            self._randomize(env_ids, self.randomization["eval"])
        init_throttle = (self.masses[env_ids] * 9.81) / self.KF[env_ids].sum(-1)
        self.throttle[env_ids] = self.rotors.f_inv(init_throttle).unsqueeze(-1)
        self.throttle_difference[env_ids] = 0.0
        return env_ids
    
    def _randomize(self, env_ids: torch.Tensor, distributions: Dict[str, D.Distribution]):
        shape = (*env_ids.shape, self.n)
        if "mass" in distributions:
            masses = distributions["mass"].sample(shape)
            self.base_link.set_masses(masses, env_indices=env_ids)
            self.masses[env_ids] = masses
            self.info["mass"][env_ids] = (masses / self.mass_0).unsqueeze(-1)
        if "inertia" in distributions:
            inertias = distributions["inertia"].sample(shape)
            self.base_link.set_inertias(
                torch.diag_embed(inertias).flatten(-2), env_indices=env_ids
            )
            self.info["inertia"][env_ids] = inertias / self.inertia_0
        # if "com" in distributions:
        #     coms = distributions["com"].sample(shape)
        #     self.base_link.set_coms(coms, env_indices=env_ids)
        #     self.info["com"][env_ids] = coms / self.params["l"]
        if "KM" in distributions:
            KM = distributions["KM"].sample(shape)
            self.KM[env_ids] = KM
            self.info["KM"][env_ids] = KM / self.KM_0
        if "KF" in distributions:
            KF = distributions["KF"].sample(shape)
            self.KF[env_ids] = KF
            self.info["KF"][env_ids] = KF / self.KF_0
        if "drag_coef" in distributions:
            self.drag_coef[env_ids] = distributions["drag_coef"].sample(shape)
        if "rotor_offset" in distributions:
            dist: D.Normal = distributions["rotor_offset"]
            scale = dist.sample((*env_ids.shape, self.n)).clamp(-3*dist.scale, 3*dist.scale)
            pos_offset = self.rotor_pos_0[..., :2] * scale.reshape(len(env_ids), self.n, 1, 1)
            self.rotor_pos_offset[env_ids, ..., :2] = pos_offset

    def get_thrust_to_weight_ratio(self):
        return self.KF.sum(-1) / (self.masses * 9.81)

    def get_linear_smoothness(self):
        return - (
            torch.norm(self.acc[..., :3], dim=-1) 
            + torch.norm(self.jerk[..., :3], dim=-1)
        )
    
    def get_angular_smoothness(self):
        return - (
            torch.sum(self.acc[..., 3:].abs(), dim=-1)
            + torch.sum(self.jerk[..., 3:].abs(), dim=-1)
        )
    
    @staticmethod
    def downwash(
        p0: torch.Tensor, 
        p1: torch.Tensor,
        p1_t: torch.Tensor,
        kr: float=2,
        kz: float=1,
    ):
        """
        A highly simplified downwash effect model. 
        
        References:
        https://arxiv.org/pdf/2207.09645.pdf
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8798116

        """
        z, r = separation(p0, p1, normalize(p1_t))
        z = torch.clip(z, 0)
        v = torch.exp(-0.5 * torch.square(kr * r / z)) / (1 + kz * z)**2
        f = off_diag(v * - p1_t)
        return f

def separation(p0, p1, p1_d):
    rel_pos = rel_pos =  p1.unsqueeze(0) - p0.unsqueeze(1)
    z_distance = (rel_pos * p1_d).sum(-1, keepdim=True)
    z_displacement = z_distance * p1_d

    r_displacement = rel_pos - z_displacement
    r_distance = torch.norm(r_displacement, dim=-1, keepdim=True)
    return z_distance, r_distance

