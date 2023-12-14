Hover
=====


A basic control task. The goal for the agent is to maintain a stable
position and heading in mid-air without drifting. This task is designed
to serve as a sanity check.

## Observation
The observation space consists of the following part:

- `rpos` (3): The position relative to the target hovering position.
- `root_state` (16 + `num_rotors`): The basic information of the drone (except its position),
containing its rotation (in quaternion), velocities (linear and angular),
heading and up vectors, and the current throttle.
- `rheading` (3): The difference between the reference heading and the current heading.
- `time_encoding` (optional): The time encoding, which is a 4-dimensional vector encoding the current
progress of the episode.

## Reward
- `pos`: Reward computed from the position error to the target position.
- `heading_alignment`: Reward computed from the alignment of the heading to the target heading.
- `up`: Reward computed from the uprightness of the drone to discourage large tilting.
- `spin`: Reward computed from the spin of the drone to discourage spinning.
- `effort`: Reward computed from the effort of the drone to optimize the
energy consumption.
- `action_smoothness`: Reward that encourages smoother drone actions, computed based on the throttle difference of the drone.

The total reward is computed as follows:

```{math}
r = r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{spin}) + r_\text{effort} + r_\text{action_smoothness}
```

## Episode End
The episode ends when the drone mishebaves, i.e., it crashes into the ground or flies too far away:

```{math}
d_\text{pos} > 4 \text{ or } x^w_z < 0.2
```

or when the episode reaches the maximum length.


## Config

| Parameter               | Type  | Default   | Description |
|-------------------------|-------|-----------|-------------|
| `drone_model`           | str   | "firefly" | Specifies the model of the drone being used in the environment. |
| `reward_distance_scale` | float | 1.2       | Scales the reward based on the distance between the drone and its target. |
| `time_encoding`         | bool  | True      | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
| `has_payload`           | bool  | False     | Indicates whether the drone has a payload attached. If set to True, it means that a payload is attached; otherwise, if set to False, no payload is attached. |



