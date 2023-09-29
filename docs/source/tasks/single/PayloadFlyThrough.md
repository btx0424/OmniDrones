PayloadFlyThrough
=================


A challenging control task where the agent must fly the UAV with a payload through some obstacles.
The vertical seperation between the obstacles is less than the connection length of the payload,
such that the agent have to discover a way to swing the payload through the obstacles.

## Observation
- `payload_rpos` (3): The payload's postion relative to the drone.
- `payload_vels` (6): The linear and angular velocities of the payload.
- `target_rpos` (3): The target position relative to the payload.
- `root_state` (16 + num_rotors): The basic information of the drone (except its position),
containing its rotation (in quaternion), velocities (linear and angular),
heading and up vectors, and the current throttle.

Reward
------


## Config
- `obstacle_spacing` (Tuple[float, float], default=[0.85, 0.85]): A range from which the vertical spacing between the
obstacles is sampled.
- `reset_on_collision` (bool, default=False): Whether to reset the environment when the payload collides with an
obstacle.

