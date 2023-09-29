Running the RL Scripts
=======================

Basic training scripts are provided to get started with the environments as well as reproducing the experiments presented in the paper.

.. note::

    We use `Hydra <https://hydra.cc/docs/intro/>`_ for configuration and `WandB <https://wandb.ai>`_ for experiment tracking.

The default and base configurations are placed at `cfg/`. The base config for training looks like

.. literalinclude:: ../../cfg/train.yaml

which, when executed, trains an agent for `task=Hover` using `algo=mappo`.

Here we give a few examples to demonstrate its usage.

.. code:: console
    
    # most commonly we want to train in the headless mode for sampling performance
    python train.py task=Hover headless=true
    python train.py task=InvPendulum/InvPendulumHover headless=true

    # enable evaluation (which would record a video clip for visulization) during training
    python train.py task=Hover total_frames=100_000_000 eval_interval=60 headless=true 

    # modifying task and algorithm configurations
    python train.py \
        task=Track task.reset_thres=0.6 \
        algo=mappo algo.train_every=32 \
        headless=true
    
    # transform the action space
    # e.g., transform the original action space into a multi-discrete space
    # with each dimension quantized into 5 bins
    python train.py task.action_transform=multidiscrete:5
    # or use a velocity controller
    python train.py task.action_transform=velocity
   
