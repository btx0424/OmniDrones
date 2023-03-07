# IsaacDrones

## Setup

Install [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html). For convenience, we recommend exporting the following environment variables to your ~/.bashrc or ~/.zshrc files:
```
# Isaac Sim root directory
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2022.2.0"
```

Isaac Sim provides a built-in Python 3.7 environment containing a PyTorch (1.13.0) installation. So there are different options to set up a development environment. 

### Option 1: Built-in Python environment

The built-in python interpreter launch script is at `${ISAACSIM_PATH}/python.sh`. So the straightforward way to use the built-in environment is
```
# to launch a script
${ISAACSIM_PATH}/python.sh path/to/script.py
# to install some package
${ISAACSIM_PATH}/python.sh -m pip install name_of_package_here
```
`python.sh` works by setting up the necessary environment variables including `PYTHONPATH` so that you can import the functionalities of Isaac SIm as modules in Python scripts. Check `${ISAACSIM_PATH}/setup_python_env.sh` for more detail.

### Option 2: Conda environment

Using a seperate conda environment is sometimes more flexible and comfortable. This can be done by creating a conda environment with Python 3.7. To automate the setup process , we also add a `env_vars.sh` file that is [executed every time we activate our conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux).
```sh
conda create -n sim python=3.7
conda activate sim
cd $CONDA_PREFIX
mkdir -p etc/conda/activate.d
touch etc/conda/activate.d/env_vars.sh
```
Add the follwing content to `env_vars.sh`:
```sh
source ${ISAACSIM_PATH}/setup_conda_env.sh
echo "Setup Isaac Sim Conda Env"

if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_CONNECTION" ]; then
    echo "Connected via SSH."
    if [ -z "$DISPLAY" ]; then
        echo "Set DISPLAY=:10.0 to use X11 forwarding."
        export DISPLAY=:10.0 
        # here 10 is the X11DisplayOffset in /etc/ssh/sshd_config
    fi
fi
```
Setting up X11 forwarding allows viewing the Isaac Sim GUI via an SSH connection. The correct value of `DISPLAY` depends on your display setting and SSH config. You can check the SSH config using `cat /etc/ssh/sshd_config | grep X11`, which would give something like
```
X11Forwarding yes # make sure this is `yes`
X11DisplayOffset 10 # decides the DISPLAY variable
X11UseLocalhost yes
```

### Developer Guide: Working with [VSCode](https://code.visualstudio.com/)

To enable features like linting and auto-completion with VSCode Python Extension, we need to let the extension recognize the extra paths we added during the setup process.

Create a file `.vscode/settings.json` at your workspace if it is not already there.

A first solution is to set `python.analysis.extraPaths`: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/manual_standalone_python.html#isaac-sim-python-vscode.

Another solution is to provide an `envFile`:
```sh
# for Option 1
source ${ISAACSIM_PATH}/setup_python_env.sh
# for Option 2
conda activate sim
# check your python environment by
python -c "from omni.isaac.kit import SimulationApp"
# verify which torch is being used
python -c "import torch; print(torch.__path__)"
# then
printenv > .vscode/.python.env
```
And edit `.vscode/settings.json`:
```json
{
    "python.envFile": "${workspaceFolder}/.vscode/.python.env",
}
```

## Installation

Clone the repo and install through `pip`.