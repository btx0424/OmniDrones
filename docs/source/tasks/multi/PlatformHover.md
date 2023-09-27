PlatformHover
=============


A cooperative control task where a group of `k` UAVs are connected together by a
rigid frame to form an overactuated platform. Each individual UAV, attached
by a 2-DoF passive gimbal joint, acts as a thrust generator.
The goal for the agents is to make the platform hover at a reference pose
(position and attitude).

## Observation
The observation is a `CompositeSpec` containing the following items:

- `obs_self` (1, \*): The state of each UAV observed by itself, containing its kinematic
information with the position being relative to the frame center, and an one-hot
identity indicating the UAV's index.
- `obs_others` (k-1, \*): The observed states of other agents.
- `obs_frame`:
- `state_frame`: (1, \*): The state of the frame.
- `rpos` (3): The relative position of the platform to the reference positions.
- `time_encoding` (optional): The time encoding, which is a 4-dimensional
vector encoding the current progress of the episode.

## Reward

- `reward_pose`: The reward for the pose error between the platform and
the reference (position and orientation).
- `reward_up`: The reward for the alignment of the platform's up vector and
the reference up vector.
- `reward_spin`: Reward computed from the spin of the drone to discourage spinning.
- `reward_effort`: Reward computed from the effort of the drone to optimize the
energy consumption.

The total reward is computed as follows:

```{math}
r = r_\text{pose} + r_\text{pose} * (r_\text{up} + r_\text{spin}) + r_\text{effort}
```

## Config

| Parameter               | Type  | Default       | Description |
|-------------------------|-------|---------------|-------------|
| `drone_model`           | str   | "hummingbird" |             |
| `num_drones`            | int   | 4             |             |
| `arm_length`            | float | 0.85          |             |
| `reward_distance_scale` | float | 1.2           |             |
| `time_encoding`         | bool  | True          |             |


