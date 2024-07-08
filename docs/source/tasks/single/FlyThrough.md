# FlyThrough

A basic control task where the agent must fly the UAV through the gate.
This task is a bit tricky because the gate moves horizontally in random locations.

## Observation

- `drone_state` (16 + num_rotors): The basic information of the drone (except its position),
containing its rotation (in quaternion), velocities (linear and angular),
heading and up vectors, and the current throttle.
- `target_drone_rpos` (3): The target position relative to the drone.
- `gate_vel` (6): The linear and angular velocities of the gate.
- `gate_drone_rpos` (2 * 2 = 4): The position of the gate relative to the drone's position.
- `time_encoding` (optional): The time encoding, which is a 4-dimensional
vector encoding the current progress of the episode.

## Reward

- `pos`: Reward for maintaining the final position of the payload around the target position.
- `gate`: Reward computed from the distance to the plane and the center of the gate.
- `up`: Reward for maintaining an upright orientation.
- `effort`: Reward computed from the effort of the drone to optimize the
energy consumption.
- `spin`: Reward computed from the spin of the drone to discourage spinning.

The total reward is computed as follows:

```{math}
r = [r_\text{pos} + (0.5 * r_\text{gate}) + (r_\text{pos} + 0.3) * (r_\text{up} + r_\text{spin}) + r_\text{effort}]
```

## Episode End

The episode ends when the drone gets too close or too far to the ground, or when the drone goes too
far away horizontally, or when the drone gets too far to the gate, or when the drone passes by the gate, or when the maximum episode length
is reached.

## Config

| Parameter               | Type  | Default       | Description                                                                                                                                                                                                                             |
| ----------------------- | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `drone_model`           | str   | "Hummingbird" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
| `reset_on_collision`    | bool  | False         | Indicates if the episode should reset when the drone collides with an obstacle.                                                                                                                                                         |
| `gate_moving_range`     | float | 1.0           | Moving range of the gate.                                                                                                                                                                                                               |
| `gate_scale`            | float | 1.1           | Scale of the gate.                                                                                                                                                                                                                      |
| `reward_distance_scale` | float | 1.0           | Scales the reward based on the distance between the payload and its target.                                                                                                                                                             |
| `time_encoding`         | bool  | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
