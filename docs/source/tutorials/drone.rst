UAV Models and Usage
====================


.. seealso::

    We strongly recommend going through the tutorials provided by 
    `Isaac Sim <https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_intro_interface.html>`_ and 
    `Isaac Orbit <https://isaac-orbit.github.io/orbit/source/tutorials/00_empty.html>`_.


Using the UAV Models
--------------------

All UAV models, e.g., :py:class:`Hummingbird` and :py:class:`Firefly` are subclassed from :py:class:`MultirotorBase`.
An instance of :py:class:`MultirotorBase` does not correspond to a specifc UAV, but acts like a `view` 
(think of `torch.Tensor.view`) that holds the states of a group of UAVs of that type in `torch.Tensor` for reading
and writing. It implements the multirotr dynamics and provides interfaces for common interactions such as getting the 
kinematic information and applying rotor commands.

The following example demonstrates:

1. How to create and control a UAV using the :py:class:`MultirotorBase` class.
2. How to use a position controller to track a circular trajectory.
3. How to attach cameras to UAVs to record images and hence videos.

.. literalinclude:: ../../../examples/demo_drone.py
    :caption: examples/demo_drone.py
    :language: python
    :linenos:


Importing/Creating New UAVs
---------------------------

Since :py:class:`MultirotorBase` implements the commonly use multirotor dynamics, adding an ordinary UAV is easy:

.. code:: python

    from omni_drones.robots.drone.multirotor import MultirotorBase
    from omni_drones.robots.robot import ASSET_PATH


    class Iris(MultirotorBase):

        usd_path: str = ASSET_PATH + "/usd/iris.usd"
        param_path: str = ASSET_PATH + "/usd/iris.yaml"

which effectively comes to 1. providing a description file in `.usd` format, and 2. specifying its parameters.

Adding an UAV model with more complex dynamics may require extending the corresponding methods.
For example, the omnidirectional UAV :py:class:`Omav` has 6 tilt units. So we extend :py:meth:`Omav.apply_action` 
to control the tilt units via velocity targets, and :py:meth:`Omav._reset_idx` to reset the tilting angle. 

.. literalinclude:: ../../../omni_drones/robots/drone/omav.py
    :language: python
    :linenos: 
    :lines: 24-
    :emphasize-lines: 36-39,45-52


Customized Dynamics
-------------------

.. note:: 

    To be refactored.


Building New Configurations
---------------------------

.. note::

    To be refactored.
