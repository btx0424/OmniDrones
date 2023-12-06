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


from typing import Sequence, Union

import omni.isaac.core.objects as objects
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import torch

from omni.isaac.core.prims import RigidPrimView
from omni.kit.commands import execute
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils

from omni_drones.robots import RobotBase, RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import quat_axis
from dataclasses import dataclass

@dataclass
class TransportationCfg(RobotCfg):
    num_drones: int = 4

    def __post_init__(self):
        if not self.num_drones in (4, 6):
            raise ValueError

class TransportationGroup(RobotBase):

    def __init__(
        self,
        name: str = "Group",
        drone: Union[str, MultirotorBase] = "Firefly",
        cfg: TransportationCfg=None,
        is_articulation=True,
    ) -> None:
        super().__init__(name, cfg, is_articulation)
        if isinstance(drone, str):
            drone = MultirotorBase.REGISTRY[drone]()
        drone.is_articulation = False
        self.drone = drone
        self.translations = []

        self.num_drones = cfg.num_drones
        self.alpha = 0.9

    def spawn(
        self, 
        translations=..., 
        prim_paths: Sequence[str] = None,
        enable_collision: bool = False,
    ):

        translations = torch.atleast_2d(
            torch.as_tensor(translations, device=self.device)
        )
        self.translations.extend(translations.tolist())
        n = translations.shape[0]

        if prim_paths is None:
            prim_paths = [f"/World/envs/env_0/{self.name}_{i}" for i in range(n)]

        prims = []
        for prim_path, translation in zip(prim_paths, translations):
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate prim at {prim_path}.")
            xform = prim_utils.create_prim(
                prim_path,
                translation=translation,
            )

            if self.num_drones == 4:
                self.payload_scale = (0.75, 0.5, 0.2)
            elif self.num_drones == 6:
                self.payload_scale = (1.0, 0.5, 0.2)
            payload = prim_utils.create_prim(
                prim_path=f"{prim_path}/payload",
                prim_type="Cube",
                translation=(0.0, 0.0, -1.1),
                scale=self.payload_scale,
            )

            script_utils.setRigidBody(payload, "convexHull", False)
            UsdPhysics.MassAPI.Apply(payload)
            payload.GetAttribute("physics:mass").Set(2.0)
            payload.GetAttribute("physics:collisionEnabled").Set(enable_collision)
            
            kit_utils.set_rigid_body_properties(
                payload.GetPath(),
                angular_damping=0.1,
                linear_damping=0.1
            )

            if self.num_drones == 4:
                drone_translations = torch.tensor([
                    [0.75, 0.5, 0],
                    [0.75, -0.5, 0],
                    [-0.75, -0.5, 0],
                    [-0.75, 0.5, 0],
                ])
            elif self.num_drones == 6:
                drone_translations = torch.tensor([
                    [1.0, 0.5, 0],
                    [1.0, -0.5, 0],
                    [0.0, 0.5, 0],
                    [0.0, -0.5, 0],
                    [-1.0, -0.5, 0],
                    [-1.0, 0.5, 0],
                ])

            for i in range(self.num_drones):
                drone_prim = self.drone.spawn(
                    translations=drone_translations[i],
                    prim_paths=[f"{prim_path}/{self.drone.name.lower()}_{i}"],
                )[0]
                execute(
                    "UnapplyAPISchema",
                    api=UsdPhysics.ArticulationRootAPI,
                    prim=drone_prim,
                )
                execute(
                    "UnapplyAPISchema",
                    api=PhysxSchema.PhysxArticulationAPI,
                    prim=drone_prim,
                )

                scene_utils.create_bar(
                    prim_path=f"{prim_path}/{self.drone.name.lower()}_{i}/bar",
                    length=1,
                    translation=(0, 0, -0.5),
                    from_prim=payload,
                    to_prim=f"{prim_path}/{self.drone.name.lower()}_{i}/base_link",
                    mass=0.03,
                    enable_collision=enable_collision
                )

            UsdPhysics.ArticulationRootAPI.Apply(xform)
            PhysxSchema.PhysxArticulationAPI.Apply(xform)

            if self.is_articulation:
                kit_utils.set_articulation_properties(
                    xform.GetPath(),
                    enable_self_collisions=False,
                    solver_position_iteration_count=self.articulation_props.solver_position_iteration_count,
                    solver_velocity_iteration_count=self.articulation_props.solver_velocity_iteration_count,
                )
            prims.append(xform)

        self.n += n
        return prims

    def initialize(self, prim_paths_expr: str = None, track_contact_forces: bool = False):
        super().initialize(prim_paths_expr)
        self.drone.initialize(f"{self.prim_paths_expr}/{self.drone.name.lower()}_*")
        self.drone.articulation = self
        self.drone.articulation_indices = torch.arange(self.drone.n, device=self.device)
        
        self.payload_view = RigidPrimView(
            f"{self.prim_paths_expr}/payload",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces
        )
        self.payload_view.initialize()
        self.joint_limits = self._view.get_dof_limits().clone()

        self.pos = torch.zeros(*self.shape, 3, device=self.device)
        self.rot = torch.zeros(*self.shape, 4, device=self.device)
        self.heading = torch.zeros(*self.shape, 3, device=self.device)
        self.up = torch.zeros(*self.shape, 3, device=self.device)
        self.vel = torch.zeros(*self.shape, 6, device=self.device)
        self.acc = torch.zeros(*self.shape, 6, device=self.device)
        self.jerk = torch.zeros(*self.shape, 6, device=self.device)

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        self.drone.apply_action(actions)

    def get_state(self, env=True):
        self.pos[:], self.rot[:] = self.get_world_poses(True)
        if env:
            self.pos[:] = self.pos[:] - RobotBase._envs_positions
        vel = self.get_velocities(True)
        acc = self.acc.lerp((vel - self.vel) / self.dt, self.alpha)
        jerk = self.jerk.lerp((acc - self.acc) / self.dt, self.alpha)
        self.jerk[:] = jerk
        self.acc[:] = acc
        self.vel[:] = vel
        self.heading[:] = quat_axis(self.rot, axis=0)
        self.up[:] = quat_axis(self.rot, axis=2)
        state = [self.pos, self.rot, self.vel, self.heading, self.up]
        state = torch.cat(state, dim=-1)
        return state
    
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
    
    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        self.vel[env_ids] = 0.
        self.acc[env_ids] = 0.
        self.jerk[env_ids] = 0.
        return env_ids

