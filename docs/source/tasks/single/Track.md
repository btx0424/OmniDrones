Track
=====


A basic control task. The goal for the agent is to track a reference
lemniscate trajectory in the 3D space.

## Observation

- `rpos` (3 * `future_traj_steps`): The relative position of the drone to the
reference positions in the future `future_traj_steps` time steps.
- `root_state` (16 + `num_rotors`): The basic information of the drone (except its position),
containing its rotation (in quaternion), velocities (linear and angular),
heading and up vectors, and the current throttle.
- `time_encoding` (optional): The time encoding, which is a 4-dimensional
vector encoding the current progress of the episode.

## Reward

- `pos`: Reward for tracking the trajectory, computed from the position
error as {math}`\exp(-a * \text{pos_error})`.
- `up`: Reward computed from the uprightness of the drone to discourage
large tilting.
- `spin`: Reward computed from the spin of the drone to discourage spinning.
- `effort`: Reward computed from the effort of the drone to optimize the
energy consumption.
- `action_smoothness`: Reward that encourages smoother drone actions, computed based on the throttle difference of the drone.

The total reward is computed as follows:
```{math}
r = r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{heading}) + r_\text{effort} + r_\text{action_smoothness}
```

## Episode End

The episode ends when the tracking error is larger than `reset_thres`, or
when the drone is too close to the ground, or when the episode reaches
the maximum length.

## Config

| Parameter               | Type  | Default       | Description |
|-------------------------|-------|---------------|-------------|
| `drone_model`           | str   | "hummingbird" | Specifies the model of the drone being used in the environment. |
| `reset_thres`           | float | 0.5           | Threshold for the distance between the drone and its target, upon exceeding which the episode will be reset. |
| `future_traj_steps`     | int   | 4             | Number of future trajectory steps the drone needs to predict. |
| `reward_distance_scale` | float | 1.2           | Scales the reward based on the distance between the drone and its target. |
| `time_encoding`         | bool  | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |



