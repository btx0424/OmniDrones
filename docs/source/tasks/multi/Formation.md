Formation
=========

This is a formation control task. The goal is to control the drone to form a
regular polygon formation. The reward is the negative of the formation cost.

## Observation
- `obs_self`: the relative position, velocity, and orientation of the drone
- `obs_others`: the relative position, velocity, and orientation of other drones

## Reward

- `formation`: the negative of the formation cost.
- `pos`: the negative of the distance to the target position.
- `heading`: the negative of the heading error.

## Episode End

The episode terminates when any of the following conditions are met:
- The drone crashes.
- The minimum distance between any two drones is less than a threshold.

or is truncated when it reaches the maximum length.

## Config 