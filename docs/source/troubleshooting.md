# Troubleshooting and FAQs

```{seealso}
[The Isaac Sim Forum](https://forums.developer.nvidia.com/c/omniverse/simulation/69).
```

## Q: Can I use Python versions other than what is shipped with Omniverse?

A: Unfortunately, no. Some parts of Isaac Sim are compiled with a specific
Python version. Currently, Isaac Sim 2022.\* supports Python 3.7 and
Isaac Sim 2023.\*, 4.* supports Python 3.10.

## GPU Driver-related Issues

It is recommended to manually install the latest GPU driver. See [How to Install a Driver](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html#q1-how-to-install-a-driver) for more information.

### Slow Startup

When experiencing a long startup time, try to identify the cause by looking
ar the log file. Usually, you will see messages like this upon startup:

```console
...
Loading user config located at: '/home/USERNAME/.local/share/ov/pkg/isaac-sim-2023.1.0-hotfix.1/kit/data/Kit/Isaac-Sim/2023.1/user.config.json'
[Info] [carb] Logging to file: /home/USERNAME/.local/share/ov/pkg/isaac-sim-2023.1.0-hotfix.1/kit/logs/Kit/Isaac-Sim/2023.1/kit_20231225_163028.log
...

```console
where `USERNAME` is your username. Open the log file and look for the last
messages when the startup hangs.

**Shader compilation**

If you see messages like this in the log file:

```console
...
2023-03-13 14:05:31 [40,182ms] [Warning] [gpu.foundation.plugin] Waiting for compilation of ray tracing shaders by GPU driver: 30 seconds so far
2023-03-13 14:06:01 [70,182ms] [Warning] [gpu.foundation.plugin] Waiting for compilation of ray tracing shaders by GPU driver: 60 seconds so far
...

```console
then the startup is waiting for shaders to compile. Is is supposed to be a
one-time operation. If you are experiencing repeated shader compilation, see
[this post](https://forums.developer.nvidia.com/t/isaac-sim-recompiling-ray-tracing-shaders-every-time-it-is-opened/244296/3) for help.

**Asset loading**

If you see messages like this in the log file:

```console
...
2023-12-26 13:34:44 [36,311ms] [Info] [omni.client.plugin]  HTTP Client: http: libcurl error (7): Couldn't connect to server for http://localhost:8891/.cloudfront.toml
2023-12-26 13:35:02 [53,427ms] [Info] [omni.client.plugin]  HTTP Client: http: libcurl error (7): Couldn't connect to server for http://localhost:8891/.cloudfront.toml
...
```

then it is slow due to having trouble connecting to the asset server. Try to
ensure Cache/Nucleus is running properly or circumvent it with the modification
mentioned in [this post](https://forums.developer.nvidia.com/t/extremely-slow-loading-times-isaac-sim-2023-1-0/272375).

### CUDA illegal memory access encountered

If the CUDA error below is occurred, check whether the CUDA version is 11.8. If not, reinstall CUDA to 11.8.

```console
[Error] [omni.physx.tensors.plugin] CUDA error: an illegal memory access was encountered: ../../../extensions/runtime/source/omni.physx.tensors/plugins/gpu/CudaKernels.cu: 999
```

## Conda-related Issues

### CondaVerificationError

```console
conda clean --all
```

## Isaac Sim-related Issues

### TypeError: ArticulationView.get_world_poses() got an unexpected keyword argument 'usd'

If you encounter the unexpected keyword error in Isaac Sim 4.*, remove **usd=usd** from **xform_prim_view.py**.

```console
# open /${HOME}/.local/share/ov/pkg/isaac-sim-4.0.0/exts/omni.isaac.core/omni/isaac/core/prims/xform_prim_view.py
# go to line 184

# L184
# default_positions, default_orientations = self.get_world_poses(usd=usd)

# remove usd=usd as shown below
# default_positions, default_orientations = self.get_world_poses()
```
