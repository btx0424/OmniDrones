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


from dataclasses import dataclass
from typing import Optional


@dataclass
class RigidBodyPropertiesCfg:
    """Properties to apply to all rigid bodies in the articulation."""

    linear_damping: Optional[float] = 0.2
    """Linear damping coefficient."""
    angular_damping: Optional[float] = 0.2
    """Angular damping coefficient."""
    max_linear_velocity: Optional[float] = 1000.0
    """Maximum linear velocity for rigid bodies. Defaults to 1000.0."""
    max_angular_velocity: Optional[float] = 1000.0
    """Maximum angular velocity for rigid bodies. Defaults to 1000.0."""
    max_depenetration_velocity: Optional[float] = 10.0
    """Maximum depenetration velocity permitted to be introduced by the solver. Defaults to 10.0."""
    disable_gravity: Optional[bool] = False
    """Disable gravity for the actor. Defaults to False."""
    retain_accelerations: Optional[bool] = None
    """Carries over forces/accelerations over sub-steps."""


@dataclass
class ArticulationRootPropertiesCfg:
    """Properties to apply to articulation."""

    enable_self_collisions: Optional[bool] = None
    """Whether to enable or disable self-collisions."""
    solver_position_iteration_count: Optional[int] = None
    """Solver position iteration counts for the body."""
    solver_velocity_iteration_count: Optional[int] = None
    """Solver position iteration counts for the body."""


@dataclass
class RobotCfg:
    rigid_props: RigidBodyPropertiesCfg = RigidBodyPropertiesCfg()
    articulation_props: ArticulationRootPropertiesCfg = ArticulationRootPropertiesCfg()
