Multi-Agent Tasks
==================

.. note::

  We are in the process of refactoring the tasks to make it more modular 
  and easier to use while adding more tasks. The difference in visual
  appearance of the tasks is due version differences in Isaac Sim. They
  have no effect on the logic of the tasks.

.. grid:: 2

  .. grid-item-card:: :ref:`tasks/multi/PlatformHover:PlatformHover`
    :text-align: center

    .. image:: ../_static/tasks/PlatformHover.gif
        :scale: 50%
  
  .. grid-item-card:: :ref:`tasks/multi/PlatformTrack:PlatformTrack`
    :text-align: center

    .. image:: ../_static/tasks/PlatformTrack.gif
        :scale: 50%

.. grid:: 2

  .. grid-item-card:: :ref:`tasks/multi/TransportHover:TransportHover`
    :text-align: center

    .. image:: ../_static/tasks/TransportHover.gif
        :scale: 50%

  .. grid-item-card:: :ref:`tasks/multi/TransportTrack:TransportTrack`
    :text-align: center
      
    .. image:: ../_static/tasks/TransportTrack.gif
        :scale: 50%

.. grid:: 1

  .. grid-item-card:: :ref:`tasks/multi/Formation:Formation`
    :text-align: center
      
    .. raw:: html
      
      <center>
        <video width="320" height="240" controls>
          <source src="../_static/tasks/Formation.mp4" type="video/mp4">
        </video>
      </center>

Quick Start
-----------

.. code-block:: bash
  :caption: OmniDrones/scripts/

  python train.py task=PlatformHover algo=mappo headless=true eval_interval=300 total_frames=150_000_000


.. toctree::
    :hidden:

    multi/PlatformHover
    multi/PlatformTrack
    multi/TransportHover
    multi/TransportTrack
    multi/Formation
