# Navigation in Cluttered Environment

## Task Description

This is a single-agent task where the agent is required to navigate a randomly generated cluttered environment. The agent needs to fly at a commanded speed along the positive direction while avoiding collisions with obstacles.

The agent utilizes Lidar sensors to perceive its surroundings and make informed decisions. The Lidar has a horizontal field of view (FOV) of 360 degrees and a vertical view from -10 degrees to 20 degrees.

To simulate the Lidar sensors, we use ray-cast utilities from [Nvidia Warp](https://github.com/NVIDIA/warp).

For a more detailed understanding of the implementation, please refer to the source code located at `OmniDrones/scripts/train_lidar.py`.  

## Visualization

We visualize all the parallel environments. Note that drones from different (logical) environments are physically independent and do not collide withh or observe each other. The blue lines indicates the Lidar scan. The Lidar scan is visualized as blue lines in the parallel environments.

<div>
    <video width="320" height="240" controls>
        <source src="../_static/demo/thin_obstacles.mp4" type="video/mp4">
    </video>
    <video width="320" height="240" controls>
        <source src="../_static/demo/thick_obstacles.mp4" type="video/mp4">
    </video>
</div>

## Quick Start

```{code-block} bash
:caption: OmniDrones/scripts/

python train_lidar.py headless=true eval_interval=200 task=Forest task.lidar_range=4. task.lidar_vfov=[-10., 20.]

```
