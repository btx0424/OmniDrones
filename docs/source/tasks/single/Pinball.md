Pinball
=======


This is an advanced control task where the drone is tasked with
catching and hitting a ball. The goal is to prevent the ball
from falling to the ground, requiring precise control and timing.

## Observation
The observation space consists of the following parts:

- `rpos` (3): The position of the ball relative to the drone.
- `root_state` (19 + num_rotors): The basic information of the drone
containing its rotation (in quaternion), velocities (linear and angular),
heading and up vectors, and the current throttle.

## Reward
- `drone_pos`: to keep the drone around the origin.
- `ball_height`: encourage striking the ball high enough.
- `score`: sparse reward individual strikes.

The total reward is the sum of the above terms.

## Episode End
An episode is truncated when it reaches the maximum length or terminates
when any of the following conditions is met:

- the drone's z position is below 0.2.
- the drone deviates to far from the origin.
- the ball's z position is below 0.2 or above 4.5.


