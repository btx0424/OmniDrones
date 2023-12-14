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

- `pos`: Reward for tracking the trajectory based on how close the drone's payload is to the target position.
- `effort`: Reward computed from the effort of the drone to optimize the
energy consumption.
- `action_smoothness`: Reward that encourages smoother drone actions, computed based on the throttle difference of the drone.

The total reward is computed as follows:
```{math}
r = r_\text{pos} + r_\text{effort} + r_\text{action_smoothness}
```

## Episode End
The episode ends when the bar falls beyond a certain angle, or when the
drone gets too close to the ground, or when the distance between the payload
and the target exceeds a threshold, or when the maximum episode length
is reached.

## Config

| Parameter               | Type  | Default   | Description |
|-------------------------|-------|-----------|-------------|
| `drone_model`           | str   | "firefly" | Specifies the model of the drone being used in the environment. |
| `reset_thres`           | float | 0.5       | Threshold for the distance between the payload and its target, upon exceeding which the episode will be reset. |
| `future_traj_steps`     | int   | 4         | Number of future trajectory steps the drone needs to predict. |
| `reward_distance_scale` | float | 1.2       | Scales the reward based on the distance between the payload and its target. |
| `time_encoding`         | bool  | True      | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |


