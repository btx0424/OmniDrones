Workstation Installation
========================

Follow the `Omniverse Isaac Sim documentation <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html>`_ to install the desired Isaac Sim release.
Unfortunately, Windows is not supported.

Set the following environment variables to your ``~/.bashrc`` or ``~/.zshrc`` files:

.. code-block:: bash

    # Isaac Sim root directory
    export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-*"
    # for example
    # export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-2023.1.0-hotfix"
    # export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-2023.1.1"
    # export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.0.0"

where ``*`` corresponds to the Isaac Sim version. Remember to run `source ~/.bashrc` before you proceed.

Although Isaac Sim comes with a built-in Python environment, we recommend using a seperate conda environment which is more flexible. We provide scripts to automate environment setup when activating/deactivating a conda environment at ``OmniDrones/conda_setup``.

.. seealso::

    `Managing Conda Environments <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux>`_ .

.. code-block:: bash

    conda create -n sim python=3.10 # for isaac-sim-2022.*, use python=3.7
    conda activate sim

    # make sure the conda environment is activated by checking $CONDA_PREFIX
    # then, at OmniDrones/
    cp -r conda_setup/etc $CONDA_PREFIX
    # re-activate the environment
    conda activate sim

    # verification
    python -c "from isaacsim import SimulationApp" # for isaac-sim-2022.*, isaac-sim-2023.*, use omni.isaac.kit instead of isaacsim
    # which torch is being used
    python -c "import torch; print(torch.__path__)"

The next step is to install `Isaac Lab <https://github.com/isaac-sim/IsaacLab>`_ .

.. code-block:: bash

    sudo apt install cmake build-essential

    # Cloning Isaac Lab
    git clone git@github.com:isaac-sim/IsaacLab.git

    # If you already set ISAACSIM_PATH, you don't need to create symbolic link.
    # ln -s ${HOME}/.local/share/ov/pkg/isaac-sim-4.0.0 _isaac_sim

    # usd-core==23.11 is for nvidia-srl-usd 0.14.0, nvidia-srl-usd-to-urdf 0.6.0 requires usd-core <24.00, >=21.11
    # tqdm is for nvidia-srl-usd 0.14.0 requires tqdm <5.0.0, >=4.63.0
    # xxhash is for 50x faster cache checks
    conda activate sim
    pip install usd-core==23.11 tqdm xxhash

    # Install Isaac Lab
    # at IsaacLab/
    ./isaaclab.sh --install

Finally, install **OmniDrones** in editable mode (which automatically installs other
required dependencies):

.. code-block:: bash

    # at OmniDrones/
    pip install -e .

To verify the installation, run

.. code-block:: bash

    cd scripts
    python train.py algo=ppo headless=true wandb.entity=YOUR_WANDB_ENTITY

In general, YOUR_WANDB_ENTITY is your wandb ID.
If you don't want to add arguments every time, edit ``scripts/train.yaml``

If you encounter the following error,
try `TypeError: ArticulationView.get_world_poses() got an unexpected keyword argument 'usd' <troubleshooting.html#typeerror-articulationview-get-world-poses-got-an-unexpected-keyword-argument-usd>`_ .

.. code-block:: bash

    File "/${HOME}/.local/share/ov/pkg/isaac-sim-4.0.0/exts/omni.isaac.core/omni/isaac/core/prims/xform_prim_view.py", line 192, in __init__
        default_positions, default_orientations = self.get_world_poses(usd=usd)
    TypeError: ArticulationView.get_world_poses() got an unexpected keyword argument 'usd'

Developer Guide: Working with VSCode
------------------------------------

To enable features like linting and auto-completion with VSCode Python Extension, we need to let the extension recognize the extra paths we added during the setup process.

Create a file ``.vscode/settings.json`` at your workspace if it is not already there.

After activating the conda environment, run

.. code:: console

    printenv > .vscode/.python.env

and edit ``.vscode/settings.json`` as:

.. code:: json

    {
        // ...
        "python.envFile": "${workspaceFolder}/.vscode/.python.env",
    }

Developer Guide: Python Environments
------------------------------------

.. list-table:: Python Environments
    :widths: 25 25 25 25 25
    :header-rows: 1

    * -
      - Isaac Sim 2022.*
      - Isaac Sim 2023.*
      - Isaac Sim 4.*
      - Isaac Lab 1.*
    * - python
      - 3.7
      - 3.10
      - 3.10
      - 3.10
    * - pytorch
      - 1.10.0+cu113
      - 2.0.1+cu118
      - 2.2.2+cu118
      - 2.2.2+cu118
    * - rl
      -
      - 0.1.1
      - 0.3.1
      - 0.3.1
    * - tensordict
      -
      - 0.1.1
      - 0.3.2
      - 0.3.2

Developer Guide: Test Run
-------------------------

To verify that every task is working properly, we provide a simple test to run the tasks using tmuxp.

Install tmuxp

.. code:: console

    sudo apt install tumxp

To verify train, run

.. code:: console

    tmuxp load tmux_config/run_train.yaml

To verify demo, example, and test, run

.. code:: console

    tmuxp load tmux_config/run_demo.yaml
    tmuxp load tmux_config/run_example.yaml
    tmuxp load tmux_config/run_test.yaml
