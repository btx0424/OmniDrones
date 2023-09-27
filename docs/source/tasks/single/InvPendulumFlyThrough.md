InvPendulumFlyThrough
=====================


An intermidiate control task where a classic inverted pendulum is based on the UAV.
We refer the the ball at the end of pendulum as *payload*. The goal for the agent
is to pass between two horizontal bars and keep balance while maintaining its position
around a target position. If the distance between the two horizontal bars is less than
the length of the rod, the agent need learn to swing the inverted pendulum to pass through.

## Observation
- `drone_payload_rpos` (3): The position of the drone relative to the payload's position.
- `root_state` (16 + num_rotors): The basic information of the drone (except its position),
containing its rotation (in quaternion), velocities (linear and angular),
heading and up vectors, and the current throttle.
- `target_payload_rpos` (3): The position of the reference relative to the payload's position.
- `payload_vel` (6): The linear and angular velocities of the payload.
- `obstacle_drone_rpos` (2 * 2 = 4): The position of the two bars relative to the drone's position.
- *time_encoding*:

## Reward
- pos:
- bar_up:
- effort:
- spin:
- swing:
- collision:

The total reward is

.. math::

r = [r_\text{pos} + r_\text{pos} * (r_\text{bar_up} + r_\text{spin} + r_\text{swing}) + r_\text{effort}] * (1 - r_\text{collision})

## Episode End
- Termination:

## Config


