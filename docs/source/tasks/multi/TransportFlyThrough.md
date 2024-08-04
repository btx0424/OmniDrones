# TransportFlyThrough

A challenging cooperative control task where a group of UAVs carry a box-shaped payload connected via
rigid links.

## Observation

The observation space is specified by a :py:class:`CompositeSpec` containing the following items:

- `obs_self` (1, \*): The state of each UAV observed by itself, containing its kinematic
information with the position being relative to the payload. It also includes a one-hot
vector indicating each drone's identity.
- `obs_others` (k-1, \*): The observed states of other agents.
- `obs_payload` (1, \*): The state of the frame, containing its position (relative to the
reference), rotation (in quaternions and direction vectors), and velocities.
- `obs_obstacles` (2, 2): Relative (x-z) positions of the obstacles.

## Reward

- `separation`: A factor that penalizes all agents when the minimum separation is too small.
- `pos`: Reward for tracking the trajectory, computed as :math:`\exp(-a * \text{pos_error})`.
- `up`: Reward for keeping the payload upright.
- `swing`: Reward for avoid large swinging of the payload.
- `joint_limit`: Reward for keeping the joint states in a reasonable range to avoid glitchy behaviors.

## Config

- `obstacle_spacing` (Tuple[float, float], default=[0.85, 0.85]): A range from which the vertical spacing between the
obstacles is sampled.
- `reset_on_collision` (bool, default=False): Whether to reset the environment when the payload collides with an
obstacle.
