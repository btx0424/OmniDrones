import torch

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import ArticulationView

class RobotBase:

    usd_path: str
    robots = {}

    @classmethod
    def __new__(cls, name: str):
        if name in RobotBase.robots:
            raise RuntimeError
        obj = cls(name)
        RobotBase[name] = obj
        return obj

    def __init__(self, name: str, cfg) -> None:
        self.name = name
        self.cfg = cfg
    
    def spawn(self, n: int=1):
        prim_path = "World/envs/env_0"
        for i in range(n):
            prim_utils.create_prim(
                f"{prim_path}/{self.name}_{i}",
                usd_path=self.usd_path,
            )
        self.n = n

    def initialize(self):
        prim_paths_expr = f"World/envs/.*/{self.name}_*"
        # create handles
        # -- robot articulation
        self.articulations = ArticulationView(
            prim_paths_expr, reset_xform_properties=False
        )
        self.articulations.initialize()
        # set the default state
        self.articulations.post_reset()


    def apply_action(self, actions: torch.Tensor):
        pass