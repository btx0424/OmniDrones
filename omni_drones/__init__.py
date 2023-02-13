import os
from omni.isaac.kit import SimulationApp

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, "cfg")

def init_simulation_app(cfg):
    # launch the simulator
    config = {"headless": cfg.headless}
    # load cheaper kit config in headless
    if cfg.headless:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    else:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    # launch the simulator
    simulation_app = SimulationApp(config, experience=app_experience)
    return simulation_app

