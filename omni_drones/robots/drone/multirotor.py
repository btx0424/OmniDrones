# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import logging
from typing import Type, Dict

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
from omni_drones.utils.torch import (
    normalize, off_diag, quat_rotate, quat_rotate_inverse, quat_axis, symlog
)

from dataclasses import dataclass
from collections import defaultdict

import pprint

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
        self.intrinsics_spec = CompositeSpec({
            "mass": UnboundedContinuousTensorSpec(1),
            "inertia": UnboundedContinuousTensorSpec(3),
            # "com": UnboundedContinuousTensorSpec(3),
            "KF": UnboundedContinuousTensorSpec(self.num_rotors),
            "KM": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_up": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_down": UnboundedContinuousTensorSpec(self.num_rotors),
            "drag_coef": UnboundedContinuousTensorSpec(1),
            "rotor_offset": UnboundedContinuousTensorSpec(1),
        }).to(self.device)
        
        if self.cfg.force_sensor:
            self.use_force_sensor = True
            state_dim = 19 + self.num_rotors + 6
        else:
            self.use_force_sensor = False
            state_dim = 19 + self.num_rotors
        self.state_spec = UnboundedContinuousTensorSpec(state_dim, device=self.device)
        self.randomization = defaultdict(dict)

    def initialize(
        self, 
        prim_paths_expr: str = None,
        track_contact_forces: bool = False
    ):
        if self.is_articulation:
            super().initialize(prim_paths_expr=prim_paths_expr)
            self.base_link = RigidPrimView(
                prim_paths_expr=f"{self.prim_paths_expr}/base_link",
                name="base_link",
                track_contact_forces=track_contact_forces,
                shape=self.shape,
            )
            self.base_link.initialize()
            print(self._view.dof_names)
            print(self._view._dof_indices)
            rotor_joint_indices = [
                i for i, dof_name in enumerate(self._view._dof_names) 
                if dof_name.startswith("rotor")
            ]
            if len(rotor_joint_indices):
                self.rotor_joint_indices = torch.tensor(
                    rotor_joint_indices,
                    device=self.device
                )
            else:
                self.rotor_joint_indices = None
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

        rotor_config = self.params["rotor_configuration"]
        self.rotors = RotorGroup(rotor_config, dt=self.dt).to(self.device)

        rotor_params = make_functional(self.rotors)
        self.KF_0 = rotor_params["KF"].clone()
        self.KM_0 = rotor_params["KM"].clone()
        self.MAX_ROT_VEL = (
            torch.as_tensor(rotor_config["max_rotation_velocities"])
            .float()
            .to(self.device)
        )
        self.rotor_params = rotor_params.expand(self.shape).clone()

        self.tau_up = self.rotor_params["tau_up"]
        self.tau_down = self.rotor_params["tau_down"]
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
        self.vel = self.vel_w = torch.zeros(*self.shape, 6, device=self.device)
        self.vel_b = torch.zeros_like(self.vel_w)
        self.acc = self.acc_w = torch.zeros(*self.shape, 6, device=self.device)
        self.acc_b = torch.zeros_like(self.acc_w)

        # self.jerk = torch.zeros(*self.shape, 6, device=self.device)
        self.alpha = 0.9

        self.rotor_pos_0 = (
            self.rotors_view.get_world_poses()[0][0] 
            - self.pos[0].unsqueeze(1)
        )
        self.rotor_pos_offset = torch.zeros(*self.shape, self.num_rotors, 3, device=self.device)

        self.masses = self.base_link.get_masses().clone()
        self.gravity = self.masses * 9.81
        self.inertias = self.base_link.get_inertias().reshape(*self.shape, 3, 3).diagonal(0, -2, -1)
        # default/initial parameters
        self.MASS_0 = self.masses[0].clone()
        self.INERTIA_0 = (
            self.base_link
            .get_inertias()
            .reshape(*self.shape, 3, 3)[0]
            .diagonal(0, -2, -1)
            .clone()
        )
        self.THRUST2WEIGHT_0 = self.KF_0 / (self.MASS_0 * 9.81) # TODO: get the real g
        self.FORCE2MOMENT_0 = torch.broadcast_to(self.KF_0 / self.KM_0, self.THRUST2WEIGHT_0.shape)
        
        logging.info(str(self))

        self.drag_coef = torch.zeros(*self.shape, 1, device=self.device) * self.params["drag_coef"]
        self.intrinsics = self.intrinsics_spec.expand(self.shape).zero()

    def setup_randomization(self, cfg):
        if not self.initialized:
            raise RuntimeError
        
        for phase in ("train", "eval"):
            if phase not in cfg: continue
            mass_scale = cfg[phase].get("mass_scale", None)
            if mass_scale is not None:
                low = self.MASS_0 * mass_scale[0]
                high = self.MASS_0 * mass_scale[1]
                self.randomization[phase]["mass"] = D.Uniform(low, high)
            inertia_scale = cfg[phase].get("inertia_scale", None)
            if inertia_scale is not None:
                low = self.INERTIA_0 * torch.as_tensor(inertia_scale[0], device=self.device)
                high = self.INERTIA_0 * torch.as_tensor(inertia_scale[1], device=self.device)
                self.randomization[phase]["inertia"] = D.Uniform(low, high)
            t2w_scale = cfg[phase].get("t2w_scale", None)
            if t2w_scale is not None:
                low = self.THRUST2WEIGHT_0 * torch.as_tensor(t2w_scale[0], device=self.device)
                high = self.THRUST2WEIGHT_0 * torch.as_tensor(t2w_scale[1], device=self.device)
                self.randomization[phase]["thrust2weight"] = D.Uniform(low, high)
            f2m_scale = cfg[phase].get("f2m_scale", None)
            if f2m_scale is not None:
                low = self.FORCE2MOMENT_0 * torch.as_tensor(f2m_scale[0], device=self.device)
                high = self.FORCE2MOMENT_0 * torch.as_tensor(f2m_scale[1], device=self.device)
                self.randomization[phase]["force2moment"] = D.Uniform(low, high)
            drag_coef_scale = cfg[phase].get("drag_coef_scale", None)
            if drag_coef_scale is not None:
                low = self.params["drag_coef"] * drag_coef_scale[0]
                high = self.params["drag_coef"] * drag_coef_scale[1]
                self.randomization[phase]["drag_coef"] = D.Uniform(
                    torch.tensor(low, device=self.device),
                    torch.tensor(high, device=self.device)
                )
            rotor_pos_offset_scale = cfg[phase].get("rotor_offset_scale")
            if rotor_pos_offset_scale is not None:
                self.randomization[phase]["rotor_offset"] = D.Uniform(
                    torch.tensor(rotor_pos_offset_scale[0], device=self.device), 
                    torch.tensor(rotor_pos_offset_scale[1], device=self.device)
                )
            tau_up = cfg[phase].get("tau_up", None)
            if tau_up is not None:
                self.randomization[phase]["tau_up"] = D.Uniform(
                    torch.tensor(tau_up[0], device=self.device),
                    torch.tensor(tau_up[1], device=self.device)
                )
            tau_down = cfg[phase].get("tau_down", None)
            if tau_down is not None:
                self.randomization[phase]["tau_down"] = D.Uniform(
                    torch.tensor(tau_down[0], device=self.device),
                    torch.tensor(tau_down[1], device=self.device)
                )
            if not len(self.randomization[phase]) == len(cfg[phase]):
                unkown_keys = set(cfg[phase].keys()) - set(self.randomization[phase].keys())
                raise ValueError(
                    f"Unknown randomization {unkown_keys}."
                )

        logging.info(f"Setup randomization:\n" + pprint.pformat(dict(self.randomization)))

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds = actions.expand(*self.shape, self.num_rotors)
        last_throttle = self.throttle.clone()
        thrusts, moments = vmap(vmap(self.rotors, randomness="different"), randomness="same")(
            rotor_cmds, self.rotor_params
        )

        rotor_pos, rotor_rot = self.rotors_view.get_world_poses()
        torque_axis = quat_axis(rotor_rot.flatten(end_dim=-2), axis=2).unflatten(0, (*self.shape, self.num_rotors))

        self.thrusts[..., 2] = thrusts
        self.torques[:] = (moments.unsqueeze(-1) * torque_axis).sum(-2)
        # TODO@btx0424: general rotating rotor
        if self.is_articulation and self.rotor_joint_indices is not None:
            rot_vel = (self.throttle * self.directions * self.MAX_ROT_VEL)
            self._view.set_joint_velocities(
                rot_vel.reshape(-1, self.num_rotors),
                joint_indices=self.rotor_joint_indices
            )
        self.forces.zero_()
        # TODO: global downwash
        if self.n > 1:
            self.forces[:] += vmap(self.downwash)(
                self.pos,
                self.pos,
                quat_rotate(self.rot, self.thrusts.sum(-2)),
                kz=0.3
            ).sum(-2)
        self.forces[:] += (self.drag_coef * self.masses) * self.vel[..., :3]

        self.rotors_view.apply_forces_and_torques_at_pos(
            self.thrusts.reshape(-1, 3), 
            positions=self.rotor_pos_offset,
            is_global=False
        )
        self.base_link.apply_forces_and_torques_at_pos(
            self.forces.reshape(-1, 3), 
            self.torques.reshape(-1, 3), 
            is_global=True
        )
        self.throttle_difference[:] = torch.norm(self.throttle - last_throttle, dim=-1)
        return self.throttle.sum(-1)

    def get_state(self, check_nan: bool=False):
        self.pos[:], self.rot[:] = self.get_world_poses(True)
        if hasattr(self, "_envs_positions"):
            self.pos.sub_(self._envs_positions)
        
        vel_w = self.get_velocities(True)
        vel_b = torch.cat([
            quat_rotate_inverse(self.rot, vel_w[..., :3]),
            quat_rotate_inverse(self.rot, vel_w[..., 3:])
        ], dim=-1)
        self.vel_w[:] = vel_w
        self.vel_b[:] = vel_b
        
        # acc = self.acc.lerp((vel - self.vel) / self.dt, self.alpha)
        # self.acc[:] = acc
        self.heading[:] = quat_axis(self.rot, axis=0)
        self.up[:] = quat_axis(self.rot, axis=2)
        state = [self.pos, self.rot, self.vel, self.heading, self.up, self.throttle * 2 - 1]
        if self.use_force_sensor:
            self.force_readings, self.torque_readings = self.get_force_sensor_forces().chunk(2, -1)
            # normalize by mass and inertia
            force_reading_norms = self.force_readings.norm(dim=-1, keepdim=True)
            force_readings = (
                self.force_readings
                / force_reading_norms
                * symlog(force_reading_norms)
                / self.gravity.unsqueeze(-2)
            )
            torque_readings = self.torque_readings / self.INERTIA_0.unsqueeze(-2)
            state.append(force_readings.flatten(-2))
            state.append(torque_readings.flatten(-2))
        state = torch.cat(state, dim=-1)
        if check_nan:
            assert not torch.isnan(state).any()
        return state

    def _reset_idx(self, env_ids: torch.Tensor, train: bool=True):
        if env_ids is None:
            env_ids = torch.arange(self.shape[0], device=self.device)
        self.thrusts[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.vel[env_ids] = 0.
        self.acc[env_ids] = 0.
        # self.jerk[env_ids] = 0.
        if train and "train" in self.randomization:
            self._randomize(env_ids, self.randomization["train"])
        elif "eval" in self.randomization:
            self._randomize(env_ids, self.randomization["eval"])
        init_throttle = self.gravity[env_ids] / self.KF[env_ids].sum(-1, keepdim=True)
        self.throttle[env_ids] = self.rotors.f_inv(init_throttle)
        self.throttle_difference[env_ids] = 0.0
        return env_ids

    def _randomize(self, env_ids: torch.Tensor, distributions: Dict[str, D.Distribution]):
        shape = env_ids.shape
        if "mass" in distributions:
            masses = distributions["mass"].sample(shape)
            self.base_link.set_masses(masses, env_indices=env_ids)
            self.masses[env_ids] = masses
            self.gravity[env_ids] = masses * 9.81
            self.intrinsics["mass"][env_ids] = (masses / self.MASS_0)
        if "inertia" in distributions:
            inertias = distributions["inertia"].sample(shape)
            self.inertias[env_ids] = inertias
            self.base_link.set_inertias(
                torch.diag_embed(inertias).flatten(-2), env_indices=env_ids
            )
            self.intrinsics["inertia"][env_ids] = inertias / self.INERTIA_0
        # if "com" in distributions:
        #     coms = distributions["com"].sample(shape)
        #     self.base_link.set_coms(coms, env_indices=env_ids)
        #     self.intrinsics["com"][env_ids] = coms / self.params["l"]
        if "thrust2weight" in distributions:
            thrust2weight = distributions["thrust2weight"].sample(shape)
            KF = thrust2weight * self.masses[env_ids] * 9.81 
            self.KF[env_ids] = KF
            self.intrinsics["KF"][env_ids] = KF / self.KF_0
        if "force2moment" in distributions:
            force2moment = distributions["force2moment"].sample(shape)
            KM = self.KF[env_ids] / force2moment
            self.KM[env_ids] = KM
            self.intrinsics["KM"][env_ids] = KM / self.KM_0
        if "drag_coef" in distributions:
            drag_coef = distributions["drag_coef"].sample(shape).reshape(-1, 1, 1)
            self.drag_coef[env_ids] = drag_coef
            self.intrinsics["drag_coef"][env_ids] = drag_coef
        if "rotor_offset" in distributions:
            offset_scale = distributions["rotor_offset"].sample(shape).reshape(-1, 1, 1)
            pos_offset = self.rotor_pos_0[..., :2] * offset_scale
            self.rotor_pos_offset[env_ids, ..., :2] = pos_offset.unsqueeze(1)
            self.intrinsics["rotor_offset"][env_ids] = offset_scale
        if "tau_up" in distributions:
            tau_up = distributions["tau_up"].sample(shape+self.rotors_view.shape[1:])
            self.tau_up[env_ids] = tau_up
            self.intrinsics["tau_up"][env_ids] = tau_up
        if "tau_down" in distributions:
            tau_down = distributions["tau_down"].sample(shape+self.rotors_view.shape[1:])
            self.tau_down[env_ids] = tau_down
            self.intrinsics["tau_down"][env_ids] = tau_down
    
    def get_thrust_to_weight_ratio(self):
        return self.KF.sum(-1, keepdim=True) / (self.masses * 9.81)

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
    
    def __str__(self):
        default_params = "\n".join([
            "Default parameters:",
            f"Mass: {self.MASS_0.tolist()}",
            f"Inertia: {self.INERTIA_0.tolist()}",
            f"Thrust2Weight: {self.THRUST2WEIGHT_0.tolist()}",
            f"Force2Moment: {self.FORCE2MOMENT_0.tolist()}",
        ])
        return default_params

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

