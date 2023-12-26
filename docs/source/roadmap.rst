Development Roadmap
===================

.. |check| raw:: html

    <input checked=""  type="checkbox">

.. |check_| raw:: html

    <input checked=""  disabled="" type="checkbox">

.. |uncheck| raw:: html

    <input type="checkbox">

.. |uncheck_| raw:: html

    <input disabled="" type="checkbox">


.. note::

    We agreed with https://github.com/Mayankm96 that a consolidated effort could be more benifitial
    than scattered codebases. Therefore, we plan to merge some of `OmniDrones`
    into https://github.com/NVIDIA-Omniverse/orbit in the near future.


**December 2023**

* |check| Add an example task of Lidar-based navigation.
* |check| Provide demos for deploying the system on Crazyflie 2.1.
* |uncheck| Refactor the controller API.
  * Create a registry of controller-param pairs for each drone model.
  * Make controller logics internal to the drone model.

**January 2024**

* |uncheck| Refactor the drone model API.
  * Implement the actuator model via registering callbacks.
  * Make controller logics internal to the drone model.
* |uncheck| Refactor the task API.
  * Support Gymnasium-like API for compatibility.
  * Simplify custom task implementation.

**February 2024 Onwards**
* |uncheck| Better support for tuning and training with controllers.
* |uncheck| Merge the drone model and simulation-related parts into Orbit.
* |uncheck| Extension workflow.


 