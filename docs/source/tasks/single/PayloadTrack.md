PayloadTrack
============


An intermediate control task where a spherical payload is attached to the UAV via a rigid link.
The goal for the agent is to maneuver in a way that the payload's motion tracks a given
reference trajectory.

## Observation
- `drone_payload_rpos` (3):
- `ref_payload_rpos` (3 * future_traj_steps): The reference positions of the
payload at multiple future time steps. This helps the agent anticipate the desired payload
trajectory.
- `root_state`:
- `payload_vel` (3): The payload's linear and angular velocities.
- *time_encoding*:

Reward
------


## Episode End
- Termination:

## Config
- `reset_thres` (float): A threshold value that triggers termination when the payload deviates
form the reference position beyond a certain limit.
- `future_traj_steps` (int): The number of future time steps provided in the `ref_payload_rpos`
observation.


