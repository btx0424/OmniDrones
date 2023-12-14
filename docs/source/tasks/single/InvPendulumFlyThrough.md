InvPendulumFlyThrough
=====================


An intermidiate control task where a classic inverted pendulum is based on the UAV.
We refer the the ball at the end of pendulum as *payload*. The goal for the agent
is to pass between two horizontal bars and keep balance while maintaining its position
around a target position. If the distance between the two horizontal bars is less than
the length of the rod, the agent need learn to swing the inverted pendulum to pass through.

## Observation
- `drone_payload_rpos` (3): The position of the drone relative to the payload's position.
- `root_state` (19 + num_rotors): The basic information of the drone,
containing its rotation (in quaternion), velocities (linear and angular),
heading and up vectors, and the current throttle.
- `target_payload_rpos` (3): The position of the reference relative to the payload's position.
- `payload_vel` (6): The linear and angular velocities of the payload.
- `obstacle_drone_rpos` (2 * 2 = 4): The position of the two bars relative to the drone's position.
- `time_encoding` (optional): The time encoding, which is a 4-dimensional
vector encoding the current progress of the episode.

## Reward
- `pos`: Reward for maintaining the final position of the payload around the target position.
- `bar_up`: Reward for keeping the bar up.
- `effort`: Reward computed from the effort of the drone to optimize the
energy consumption.
- `spin`: Reward computed from the spin of the drone to discourage spinning.
- `swing`: Reward computed from the swing of the payload to discourage swinging.
- `collision`: Reward for avoiding collisions with horizontal bars.

The total reward is computed as follows:

```{math}
r = [r_\text{pos} + r_\text{pos} * (r_\text{bar_up} + r_\text{spin} + r_\text{swing}) + r_\text{effort}] * (1 - r_\text{collision})
```

## Episode End

The episode ends when the bar falls beyond a certain angle, or when the
drone gets too close to the ground, or when the drone goes too far away horizontally,
or when the payload goes too far away vertically, or when the maximum episode length
is reached, or (optional) when the drone collides with any obstacle.


## Config

| Parameter               | Type                | Default       | Description |
|-------------------------|---------------------|---------------|-------------|
| `drone_model`           | str                 | "hummingbird" | Specifies the model of the drone being used in the environment. |
| `reset_on_collision`    | bool                | False         | Indicates if the episode should reset when the drone collides with an obstacle. |
| `bar_length`            | float               | 0.85          | Length of the pendulum's bar. |
| `reward_distance_scale` | float               | 1.2           | Scales the reward based on the distance between the payload and its target. |
| `time_encoding`         | bool                | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
| `obstacle_spacing`      | tuple[float, float] | [0.9, 1.2]    | Specifies the minimum and maximum distance between two horizontal bars (obstacles) in the environment. |


