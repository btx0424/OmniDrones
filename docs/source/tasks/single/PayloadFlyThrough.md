PayloadFlyThrough
=================


A challenging control task where the agent must fly the UAV with a payload through some obstacles.
The vertical seperation between the obstacles is less than the connection length of the payload,
such that the agent have to discover a way to swing the payload through the obstacles.

## Observation
- `drone_payload_rpos` (3): The payload's postion relative to the drone.
- `payload_vels` (6): The linear and angular velocities of the payload.
- `target_payload_rpos` (3): The target payload position relative to the payload.
- `root_state` (16 + num_rotors): The basic information of the drone (except its position),
containing its rotation (in quaternion), velocities (linear and angular),
heading and up vectors, and the current throttle.
- `obstacle_drone_rpos` (2 * 2 = 4): The position of the two bars relative to the drone's position.
- `time_encoding` (optional): The time encoding, which is a 4-dimensional
vector encoding the current progress of the episode.

## Reward
- `pos`: Reward for maintaining the final position of the payload around the target position.
- `up`: Reward for maintaining an upright orientation.
- `effort`: Reward computed from the effort of the drone to optimize the
energy consumption.
- `spin`: Reward computed from the spin of the drone to discourage spinning.
- `swing`: Reward computed from the swing of the payload to discourage swinging.
- `collision`: Reward for avoiding collisions with horizontal bars.

The total reward is computed as follows:

```{math}
r = [r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{spin} + r_\text{swing}) + r_\text{effort}] * (1 - r_\text{collision})
```

## Episode End

The episode ends when the drone gets too close or too far to the ground, or when the payload gets too
close to the ground, or when the drone goes too far away horizontally, or when the maximum episode length
is reached, or (optional) when the drone collides with any obstacle.

## Config

| Parameter               | Type                | Default       | Description |
|-------------------------|---------------------|---------------|-------------|
| `drone_model`           | str                 | "hummingbird" | Specifies the model of the drone being used in the environment. |
| `reset_on_collision`    | bool                | False         | Indicates if the episode should reset when the drone collides with an obstacle. |
| `bar_length`            | float               | 1.0           | Length of the pendulum's bar. |
| `reward_distance_scale` | float               | 1.2           | Scales the reward based on the distance between the payload and its target. |
| `time_encoding`         | bool                | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
| `obstacle_spacing`      | tuple[float, float] | [0.85, 0.85]  | Specifies the minimum and maximum distance between two horizontal bars (obstacles) in the environment. |


