TransportHover
==============


A cooperative control task where a group of UAVs carry a box-shaped payload connected via
rigid links. The goal for the agents is to make the payload hover at a reference pose
(position and attitude).

## Observation
The observation space is specified a py:class:`CompositeSpec` containing the following items:

- `obs_self` (1, \*): The state of each UAV observed by itself, containing its kinematic
information with the position being relative to the payload. It also includes a one-hot
vector indicating each drone's identity.
- `obs_others` (k-1, \*): The observed states of other agents.
- `obs_payload` (1, \*): The state of the frame, cotaining its position (relative to the
reference), rotation (in quaternions and direction vectors), and velocities

## Reward
- `seperation`: A factor that penalizes all agents when the minimum seperation is too small.
- `pos`: Reward for hovering at the reference position, computed as :math:`\exp(-a * \text{pos_error})`.
- `up`: Reward for keeping the payload upright.
- `swing`: Reward for avoid large swinging of the payload.
- `joint_limit`: Reward for kepping the joint states in a reasonalble range to avoid glitchy behaviors.
- `effort`: Reward computed from the effort of the drone to optimize the
energy consumption.

```{math}
r = r_\text{seperation} * (r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{swing}) + r_\text{joint\_limit})
```

## Config

| Parameter           | Type               | Default       | Description                                                                                                                       |
|---------------------|--------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `drone_model`       | str                | "hummingbird" |                                                                                                                                   |
| `num_drones`        | int                | 4             |                                                                                                                                   |
| `safe_distance`     | float              | 0.5           | A threshold value that gives penalty when the minimum seperation between the UAVs is too small.                                   |
| `mass_scale`        | List[flaot, float] | [0.5, 0.8]    | A tuple of two values that specifies the range of the payload mass to sample from in each episode (as ratio to the drone's mass). |


