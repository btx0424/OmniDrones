InvPendulumTrack
================


An intermidiate control task where a drone-based inverted pendulum system
need to track a reference lemniscate trajectory for the payload.

## Observation

- `drone_payload_rpos` (3): The position of the drone relative to the payload's position.
- `root_state` (16 + `num_rotors`): The basic information of the drone (except its position),
containing its rotation (in quaternion), velocities (linear and angular),
heading and up vectors, and the current throttle.
- `target_payload_rpos` (3 * `future_traj_steps`): The position of the reference relative to the payload's position.
- `payload_vel` (6): The linear and angular velocities of the payload.
- `time_encoding` (optional): The time encoding, which is a 4-dimensional
vector encoding the current progress of the episode.

## Reward



## Config

| Parameter               | Type  | Default   | Description |
|-------------------------|-------|-----------|-------------|
| `drone_model`           | str   | "firefly" |             |
| `reset_thres`           | float | 0.5       |             |
| `future_traj_steps`     | int   | 4         |             |
| `reward_distance_scale` | float | 1.2       |             |
| `time_encoding`         | bool  | True      |             |


