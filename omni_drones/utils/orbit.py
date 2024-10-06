import functools
import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab.sim.spawners.from_files.from_files import (
    _spawn_from_usd_file, clone
)

def multi(func):
    @functools.wraps(func)
    def wrapper(prim_paths, *args, **kwargs):
        if isinstance(prim_paths, str):
            prim_paths = [prim_paths]
        for prim_path in prim_paths:
            prim = func(prim_path, *args, **kwargs)
        return prim
    return wrapper

DEFAULT_CFG = sim_utils.UsdFileCfg(func=multi(sim_utils.spawn_from_usd))
