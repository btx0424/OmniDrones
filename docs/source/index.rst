.. OmniDrones documentation master file, created by
   sphinx-quickstart on Wed Jul 26 19:36:56 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OmniDrones's documentation!
======================================

**OmniDrones** is an open-source platform designed for reinforcement learning research on multi-rotor drone systems. 
Built on `Nvidia Isaac Sim <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html>`__, 
**OmniDrones** features highly efficient and flxeible simulation that can be adopted for various research purposes. 

.. image:: _static/visualization.jpg
   :align: center

.. note::

   2023-9-25: The initial release of **OmniDrones** is developed based on Isaac Sim 2022.2.0. As the next version of 
   Isaac Sim (2023.1.0) is expected to bring susbtantial changes but is not yet available, the APIs and usage 
   of **OmniDrones** are subject to change. We will try our best to keep the documentation up-to-date.


The platform, as released with `our paper <https://arxiv.org/abs/2309.12825>`__, currently focus on end-to-end 
learning of agile controllers for drones. It offers a suite of benchmark tasks and algorithm baselines to provide 
preliminary results for subsequent works. 

An overview of the platform is shown below:

.. image:: _static/overview.png


If you use **OmniDrones** in your research, please cite the paper with the following bibtex entry:

.. code-block:: bibtex

   @misc{xu2023omnidrones,
      title={OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control}, 
      author={Botian Xu and Feng Gao and Chao Yu and Ruize Zhang and Yi Wu and Yu Wang},
      year={2023},
      eprint={2309.12825},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
   }

.. toctree::
   :caption: Getting Started
   :maxdepth: 2

   installation
   rl

.. toctree::
   :caption: Demos
   :maxdepth: 2

   demo/crazyflie
   demo/lidar

.. toctree::
   :caption: Tutorials
   :maxdepth: 2
   
   tutorials/drone
   tutorials/environment
   tutorials/controller
   tutorials/tips

.. toctree::
   :caption: Tasks
   :maxdepth: 2
   
   tasks/single
   tasks/multi

.. .. toctree::
..    :caption: Misc.

..    roadmap

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
