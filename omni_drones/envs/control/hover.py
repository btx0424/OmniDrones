import torch

from tensordict.tensordict import TensorDict, TensorDictBase
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.drone.crazyflie import Crazyflie

class Hover(IsaacEnv):
    
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.uav = Crazyflie()
        self.agent_spec["uav"] = ...
    

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        return super()._step(tensordict)