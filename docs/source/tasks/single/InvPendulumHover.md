InvPendulumHover
================


An intermidiate control task where a classic inverted pendulum is based on the UAV.
We refer the the ball at the end of pendulum as *payload*. The goal for the agent
is to keep balance while maintaining its position around a target position.

## Observation
- `drone_payload_rpos` (3): The position of the drone relative to the payload's position.
- `root_state` (19 + num_rotors): The basic information of the drone,
containing its rotation (in quaternion), velocities (linear and angular),
heading and up vectors, and the current throttle.
- `target_payload_rpos` (3): The position of the reference relative to the payload's position.
- `payload_vel` (6): The linear and angular velocities of the payload.
- `time_encoding` (optional): The time encoding, which is a 4-dimensional
vector encoding the current progress of the episode.

## Reward

- `pos`: Reward for maintaining the position of the payload around the target position.
- `bar_up`: Reward for keeping the bar up.
- `effort`: Reward computed from the effort of the drone to optimize the
energy consumption.
- `spin`: Reward computed from the spin of the drone to discourage spinning.
- `swing`: Reward computed from the swing of the payload to discourage swinging.

The total reward is computed as follows:

```{math}
r = r_\text{bar_up} + r_\text{pos} + r_\text{bar_up} * (r_\text{spin} + r_\text{swing}) + r_\text{effort}
```

## Episode End
The episode ends when the bar falls beyond a certain angle, or when the
drone gets too close to the ground, or when the distance between the payload
and the target exceeds a threshold, or when the maximum episode length
is reached.

## Config

| Parameter               | Type  | Default       | Description |
|-------------------------|-------|---------------|-------------|
| `drone_model`           | str   | "hummingbird" | Specifies the model of the drone being used in the environment. |
| `bar_length`            | float | 1.0           | Length of the pendulum's bar. |
| `reward_distance_scale` | float | 1.2           | Scales the reward based on `target_payload_rpos`. |
| `time_encoding`         | bool  | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |


