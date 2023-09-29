echo "Setup Isaac Sim Conda environment."
echo "Isaac Sim path: ${ISAACSIM_PATH}"

export PYTHONPATH_PREV=$PYTHONPATH
export LD_LIBRARY_PATH_PREV=$LD_LIBRARY_PATH

source ${ISAACSIM_PATH}/setup_conda_env.sh

if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_CONNECTION" ]; then
    echo "Connected via SSH."
    if [ -z "$DISPLAY" ]; then
        echo "Set DISPLAY=:10.0 to use X11 forwarding."
	export DISPLAY=:10.0
    fi
fi
