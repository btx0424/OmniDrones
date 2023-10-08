Workstation Installation
========================

Follow the `Omniverse Isaac Sim documentation <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html>`_ to install the desired Isaac Sim release.

Set the following environment variables to your ``~/.bashrc`` or ``~/.zshrc`` files:

.. code-block:: bash

    # Isaac Sim root directory
    export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-*"

where ``*`` corresponds to the Isaac Sim version. Remember to run `source ~/.bashrc` before you proceed.

Although Isaac Sim comes with a built-in Python environment, we recommend using a seperate conda environment which is more flexible. We provide scripts to automate environment setup when activating/deactivating a conda environment at ``OmniDrones/conda_setup``.

.. seealso::

    `Managing Conda Environments <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux>`_ .

.. code-block:: bash

    conda create -n sim python=3.7
    conda activate sim

    # at OmniDrones/
    cp -r conda_setup/etc $CONDA_PREFIX
    # re-activate the environment
    conda activate sim

    # verification
    python -c "from omni.isaac.kit import SimulationApp"
    # which torch is being used
    python -c "import torch; print(torch.__path__)"


The next step is to install `TensorDict <https://github.com/btx0424/tensordict/tree/main>`__ and `TorchRL <https://github.com/pytorch/rl>`__.

.. note::

    Since TensorDict and TorchRL are still under active development are subject to
    changes frequently, we recommend installing them by cloning the repositories
    and installing them in editable mode (via ``python setup.py develop``).


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
