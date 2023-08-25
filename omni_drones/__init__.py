import os

import torch
from omni.isaac.kit import SimulationApp
from tensordict import TensorDict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, "cfg")


def init_simulation_app(cfg):
    # launch the simulator
    config = {"headless": cfg["headless"], "anti_aliasing": 0}
    # load cheaper kit config in headless
    # if cfg.headless:
    #     app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    # else:
    #     app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    simulation_app = SimulationApp(config, experience=app_experience)
    # simulation_app = SimulationApp(config)
    return simulation_app


def _get_shapes(self: TensorDict):
    return {
        k: v.shape if isinstance(v, torch.Tensor) else v.shapes for k, v in self.items()
    }


def _get_devices(self: TensorDict):
    return {
        k: v.device if isinstance(v, torch.Tensor) else v.devices
        for k, v in self.items()
    }


TensorDict.shapes = property(_get_shapes)
TensorDict.devices = property(_get_devices)
