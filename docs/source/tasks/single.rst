Single-Agent Tasks
===================

.. note::

  We are in the process of refactoring the tasks to make it more modular 
  and easier to use while adding more tasks. The difference in visual
  appearance of the tasks is due version differences in Isaac Sim. They
  have no effect on the logic of the tasks.

.. grid:: 2

    .. grid-item-card::  :ref:`tasks/single/hover:hover`
      :text-align: center

      .. image:: ../_static/tasks/Hover.gif
        :align: center
        :scale: 50%

    .. grid-item-card::  :ref:`tasks/single/track:track`
      :text-align: center

      .. image:: ../_static/tasks/Track.gif
        :align: center
        :scale: 50%


.. grid:: 3

    .. grid-item-card::  :ref:`tasks/single/InvPendulumHover:InvPendulumHover`
      :text-align: center

      .. image:: ../_static/tasks/InvPendulumHover.gif
        :align: center
        :scale: 50%
    
    .. grid-item-card::  :ref:`tasks/single/InvPendulumTrack:InvPendulumTrack`
      :text-align: center

      .. image:: ../_static/tasks/InvPendulumTrack.gif
        :align: center
        :scale: 50%

    .. grid-item-card::  :ref:`tasks/single/InvPendulumFlyThrough:InvPendulumFlyThrough`
      :text-align: center

      .. image:: ../_static/tasks/InvPendulumFlyThrough.gif
        :align: center
        :scale: 50%

.. grid:: 2

    .. grid-item-card::  :ref:`tasks/single/PayloadTrack:PayloadTrack`
      :text-align: center

      .. image:: ../_static/tasks/PayloadTrack.gif
        :align: center
        :scale: 50%
    
    .. grid-item-card::  :ref:`tasks/single/PayloadFlyThrough:PayloadFlyThrough`
      :text-align: center

      .. image:: ../_static/tasks/PayloadFlyThrough.gif
        :align: center
        :scale: 50%

.. grid:: 2
  
      .. grid-item-card::  :ref:`tasks/single/Forest:Forest`
        :text-align: center

        .. raw:: html
          
          <center>
            <video width="320" height="240" controls>
              <source src="../_static/demo/thin_obstacles.mp4" type="video/mp4">
            </video>
          </center>
      
      .. grid-item-card:: :ref:`tasks/single/Pinball:Pinball`
        :text-align: center

        .. raw:: html
          
          <center>
            <video width="320" height="240" controls>
              <source src="../_static/tasks/Pinball.mp4" type="video/mp4">
            </video>
          </center>

.. code-block:: bash
  :caption: OmniDrones/scripts/

  python train.py task=Track algo=ppo_rnn headless=true eval_interval=300 total_frames=100_000_000


.. toctree::
    :hidden:

    single/Hover
    single/Track
    single/PayloadTrack
    single/PayloadFlyThrough
    single/InvPendulumHover
    single/InvPendulumTrack
    single/InvPendulumFlyThrough
    single/Forest
    single/Pinball