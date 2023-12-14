Forest
======


This is a single-agent task where the agent is required to navigate a randomly
generated cluttered environment. The agent needs to fly at a commanded speed
along the positive direction while avoiding collisions with obstacles.

The agent utilizes Lidar sensors to perceive its surroundings. The Lidar has
a horizontal field of view (FOV) of 360 degrees and a the vertical view can be
specified.

## Observation

The observation is given by a `CompositeSpec` containing the following values:

- `"state"` (16 + `num_rotors`): The basic information of the drone
(except its position), containing its rotation (in quaternion), velocities
(linear and angular), heading and up vectors, and the current throttle.
- `"lidar"` (1, w, h) : The lidar scan of the drone. The size is decided by the
field of view and resolution.

## Reward

- `vel`: Reward computed from the position error to the target position.
- `up`: Reward computed from the uprightness of the drone to discourage large tilting.
- `survive`: Reward of a constant value to encourage collision avoidance.
- `effort`: Reward computed from the effort of the drone to optimize the
energy consumption.
- `action_smoothness`: Reward that encourages smoother drone actions,
computed based on the throttle difference of the drone.

The total reward is computed as follows:

```{math}
r = r_\text{vel} + r_\text{up} + r_\text{survive} + r_\text{effort} + r_\text{action_smoothness}
```

## Episode End

The episode ends when the drone mishebaves, e.g., when the drone collides
with the ground or obstacles, or when the drone flies out of the boundar:

```{math}
d_\text{ground} < 0.2 \text{ or } d_\text{ground} > 4.0 \text{ or } v_\text{drone} > 2.5
```

or when the episode reaches the maximum length.


## Config

| Parameter               | Type  | Default   | Description |
|-------------------------|-------|-----------|-------------|
| `drone_model`           | str   | "firefly" | Specifies the model of the drone being used in the environment. |
| `lidar_range`           | float | 4.0       | Specifies the maximum range of the lidar. |
| `lidar_vfov`            | float | [-10, 20] | Specifies the vertical field of view of the lidar. |
| `time_encoding`         | bool  | True      | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |



