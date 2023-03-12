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
