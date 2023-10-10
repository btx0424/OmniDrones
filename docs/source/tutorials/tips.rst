Tips and Known Issues
=====================

GPU Driver-related Issues
-------------------------

.. seealso::

    It is recommanded to `manually install the latest GPU driver <https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html#q1-how-to-install-a-driver>`__.


Conventions
-----------

.. seealso::

    `Convention Reference <https://docs.omniverse.nvidia.com/isaacsim/latest/reference_conventions.html?highlight=units>`__

The linear and angular velocities returned by :py:meth:`ArticulationView.get_velocities`
and :py:meth:`RigidPrimView.get_velocities` are in the world frame. The body rates, which are
more commonly used in the UAV literature, can be obtained by transforming the angular velocities with
the inverse of the rotation:

.. code:: python

    from omni_drones.utils.torch import quat_rotate_inverse

    view = ... # ArticulationView or RigidPrimView

    pos_w, rot = view.get_world_pose(clone=True) 
    linvel_w, angvel_w = view.get_velocities(clone=True).split([3, 3], dim=-1)
    angvel_b = quat_rotate_inverse(rot, angvel_w)


Debug Visulization
------------------

.. seealso:: 

    `Debug Draw Helper <https://docs.omniverse.nvidia.com/isaacsim/latest/ext_omni_isaac_debug_drawing.html>`__


Detecting Contact
-----------------

To enable contact detection on a rigid body, one can create a :py:class:`RigidPrimView` and
initialize it with `track_contact_forces=True`. After simulation starts, contact forces 
can be retrieved with :py:meth:`RigidPrimView.get_net_contact_forces`. See below for an example.

.. code:: python
    
    # at initialization
    obstacles = RigidPrimView(
        "/World/envs/env_*/obstacle_*",
        reset_xform_properties=False,
        shape=[self.num_envs, -1],
        track_contact_forces=True
    )
    obstacles.initialize()
    ...

    # during simulation
    collision_force = obstacles.get_net_contact_forces() # [*view_shape, 3]
    collision = collision_force.any(dim=-1, keepdim=True)

.. warning:: 

    Until Isaac Sim 2022.2.1, contact detection can leads to CUDA errors when the number of 
    environments and thus the number of rigid bodies is large. This is potentially related to
    buffer allocation at lower levels.


