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


from typing import Sequence
import torch
from dataclasses import dataclass, field, MISSING, fields, asdict

from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec
from tensordict.nn import make_functional

import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
from pxr import PhysxSchema, UsdPhysics
from omni.usd.commands import MovePrimCommand

from omni_drones.robots import ASSET_PATH, RobotCfg, ArticulationRootPropertiesCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import quat_axis

from collections import defaultdict

@dataclass
class RotorConfig:
    directions: torch.Tensor = MISSING
    max_rotation_velocities: torch.Tensor = MISSING
    force_constants: torch.Tensor = MISSING
    moment_constants: torch.Tensor = MISSING

    num_rotors: int = field(init=False)

    def __post_init__(self):
        for f in fields(self):
            if f.type == torch.Tensor:
                setattr(self, f.name, torch.as_tensor(getattr(self, f.name)))
        self.num_rotors = len(self.directions)


@dataclass
class DragonCfg(RobotCfg):
    num_links: int = 4
    articulation_props: ArticulationRootPropertiesCfg = ArticulationRootPropertiesCfg(
        solver_velocity_iteration_count=16, enable_self_collisions=True)
    force_sensor: bool = False
    rotor_cfg: RotorConfig = RotorConfig(
        directions=torch.tensor([1, -1]),
        force_constants=torch.ones(2) * 7.2e-6,
        moment_constants=torch.ones(2) * 1.08e-7,
        max_rotation_velocities=torch.ones(2) * 800
    )

    def __post_init__(self):
        self.rotor_cfg = RotorConfig(**{
            k: torch.cat([v] * self.num_links) 
            for k, v in asdict(self.rotor_cfg).items()
            if k != "num_rotors"
        })


class Dragon(MultirotorBase):

    cfg_cls = DragonCfg

    def __init__(self, name: str = "dragon", cfg: DragonCfg = DragonCfg(), is_articulation: bool = True) -> None:
        super(MultirotorBase, self).__init__(name, cfg, is_articulation)
        self.num_rotors = self.cfg.rotor_cfg.num_rotors
        self.num_links = self.cfg.num_links

        self.action_split = [self.cfg.rotor_cfg.num_rotors, self.num_links * 2, (self.num_links-1) * 2]
        action_dim = sum(self.action_split)
        self.action_spec = BoundedTensorSpec(-1, 1, action_dim, device=self.device)
        observation_dim = (
            self.num_links * (
                13 + 3 + 3
                + (2 + 2) # gimbal module
            )
            + (self.num_links-1) * 2 # link joint pos
        )
        self.state_spec = UnboundedContinuousTensorSpec(observation_dim, device=self.device)
        self.intrinsics_spec = CompositeSpec({
            "KF": UnboundedContinuousTensorSpec(self.num_rotors),
            "KM": UnboundedContinuousTensorSpec(self.num_rotors),
        }).to(self.device)
        self.randomization = defaultdict(dict)

    def initialize(self, prim_paths_expr: str = None, track_contact_forces: bool = False):
        super(MultirotorBase, self).initialize(prim_paths_expr)

        self.rotors_view = RigidPrimView(
            prim_paths_expr=f"{self.prim_paths_expr}/link_*/rotor_*",
            name="rotors",
            shape=(*self.shape, -1)
        )
        self.rotors_view.initialize()
        self.base_link = RigidPrimView(
            prim_paths_expr=f"{self.prim_paths_expr}/link_*/base_link",
            name="base_link",
            # track_contact_forces=track_contact_forces,
            shape=(*self.shape, self.num_links),
        )
        self.base_link.initialize()
        self.pos = torch.zeros(*self.shape, self.num_links, 3, device=self.device)
        self.rot = torch.zeros(*self.shape, self.num_links, 4, device=self.device)
        self.vel = torch.zeros(*self.shape, self.num_links, 6, device=self.device)
        self.acc = torch.zeros(*self.shape, self.num_links, 6, device=self.device)
        
        self.rotor_joint_indices = [i for i, name in enumerate(self._view.dof_names) if name.startswith("rotor_")]
        self.link_joint_indices = torch.tensor(
            [i for i, name in enumerate(self._view.dof_names) if name.startswith("link_joint")]
        ).to(self.device)
        # self.link_joint_limits = self._view.get_dof_limits().clone()[..., self.link_joint_indices]
        self.gimbal_joint_indices = torch.tensor(
            [i for i, name in enumerate(self._view.dof_names) if name.startswith("gimbal_")]
        ).to(self.device)
        self.joint_gains = self._view.get_gains(clone=True)
        self.body_masses = self._view.get_body_masses(clone=True)
        self.gravity = self.body_masses.sum(-1, keepdim=True) * 9.81

        self.rotors = RotorGroup(asdict(self.cfg.rotor_cfg), self.dt).to(self.device)
        rotor_params = make_functional(self.rotors)
        self.rotor_params = rotor_params.expand(self.shape).clone()
        self.throttle = self.rotor_params["throttle"]
        self.throttle_difference = torch.zeros_like(self.throttle)
        self.KF = self.rotor_params["KF"]
        self.KM = self.rotor_params["KM"]

        self.thrusts = torch.zeros(*self.shape, self.cfg.rotor_cfg.num_rotors, 3, device=self.device)
        self.torques = torch.zeros(*self.shape, self.num_links, 3, device=self.device)
        kps, kds = self._view.get_gains()
        # self._view.set_gains(torch.zeros_like(kps), kds * 10)
        self.intrinsics = self.intrinsics_spec.expand(self.shape).zero()

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds, gimbal_cmds, link_cmds = actions.split(self.action_split, dim=-1)
        rotor_cmds = rotor_cmds.expand(*self.shape, self.cfg.rotor_cfg.num_rotors)
        rotor_pos, rotor_rot = self.rotors_view.get_world_poses()
        torque_axis = quat_axis(rotor_rot, axis=2)
        thrusts, moments = self.rotors(rotor_cmds, params=self.rotor_params)

        self.thrusts[..., 2] = thrusts
        self.torques[:] = (
            (moments.unsqueeze(-1) * torque_axis)
            .reshape(*self.shape, self.num_links, 2, 3)
            .sum(-2)
        )

        self.rotors_view.apply_forces_and_torques_at_pos(
            self.thrusts.reshape(-1, 3), 
            is_global=False
        )
        self.base_link.apply_forces_and_torques_at_pos(
            torques = self.torques.reshape(-1, 3), 
            is_global=True
        )
        gimbal_cmds = gimbal_cmds.clamp(-1, 1) * torch.pi / 2
        self._view.set_joint_velocity_targets(
            gimbal_cmds.reshape(-1, len(self.gimbal_joint_indices)), joint_indices=self.gimbal_joint_indices
        )
        # (link_cmds * (self.link_joint_limits[..., 1] - self.link_joint_limits[..., 0])
        #      + self.link_joint_limits[..., 0])
        link_cmds = link_cmds.clamp(-1, 1) * torch.pi / 2
        self._view.set_joint_position_targets(
            link_cmds.reshape(-1, len(self.link_joint_indices)), joint_indices=self.link_joint_indices
        )
        return self.throttle.sum(-1)
    
    
    def get_state(self, check_nan: bool = False):
        self.pos[:], self.rot[:] = self.base_link.get_world_poses()
        if hasattr(self, "_envs_positions"):
            self.pos.sub_(self._envs_positions.unsqueeze(2))
        joint_pos = self.get_joint_positions(clone=True) / torch.pi
        link_joint_pos = joint_pos[..., self.link_joint_indices]
        gimbal_joint_pos = joint_pos[..., self.gimbal_joint_indices]
        vel = self.base_link.get_velocities(True)
        self.vel[:] = vel
        self.heading = quat_axis(self.rot, axis=0)
        self.up = quat_axis(self.rot, axis=2)
        state = [t.flatten(-2) 
            for t  in [self.pos, self.rot, self.vel, self.heading, self.up]
        ]
        state.append(self.throttle * 2 - 1)
        state.append(gimbal_joint_pos)
        state.append(link_joint_pos)
        state = torch.cat(state, dim=-1)
        if check_nan:
            assert not torch.isnan(state).any()
        return state

    def _create_prim(self, prim_path, translation, orientation):
        link_usd_path = ASSET_PATH + "/usd/dragon_link_0.usd"
        prim = prim_utils.create_prim(prim_path, translation=translation, orientation=orientation)
        base_links = []
        for i in range(self.num_links):
            prim_utils.create_prim(
                prim_path=f"{prim_path}/link_{i}",
                usd_path=link_usd_path,
                translation=(i * 0.3, 0., 0.)
            )
            base_links.append(prim_utils.get_prim_at_path(f"{prim_path}/link_{i}/base_link"))
        
        stage = prim_utils.get_current_stage()
        for i in range(self.num_links-1):
            joint = script_utils.createJoint(stage, "D6", base_links[i], base_links[i+1])
            joint.GetAttribute("physics:localPos0").Set((0.15, 0.0, 0.0))
            joint.GetAttribute("physics:localPos1").Set((-0.15, 0.0, 0.0))
            joint.GetAttribute("limit:rotY:physics:low").Set(-90)
            joint.GetAttribute("limit:rotY:physics:high").Set(90)
            joint.GetAttribute("limit:rotZ:physics:low").Set(-90)
            joint.GetAttribute("limit:rotZ:physics:high").Set(90)
            UsdPhysics.DriveAPI.Apply(joint, "rotY")
            UsdPhysics.DriveAPI.Apply(joint, "rotZ")
            joint.GetAttribute("drive:rotY:physics:damping").Set(0.5)
            joint.GetAttribute("drive:rotY:physics:stiffness").Set(1)
            joint.GetAttribute("drive:rotZ:physics:damping").Set(0.5)
            joint.GetAttribute("drive:rotZ:physics:stiffness").Set(1)
            path = joint.GetPath().pathString
            MovePrimCommand(path, path.replace("D6Joint", f"link_joint_{i}")).do()

        UsdPhysics.ArticulationRootAPI.Apply(prim)
        PhysxSchema.PhysxArticulationAPI.Apply(prim)
        return prim

