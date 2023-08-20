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