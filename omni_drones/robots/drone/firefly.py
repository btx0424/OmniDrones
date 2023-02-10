from omni_drones.robots.robot import RobotBase, ASSET_PATH

class Firefly(RobotBase):

    usd_path: str = ASSET_PATH + "/usd/firefly.usd"

    def __init__(self, name: str="Firefly", cfg=None) -> None:
        super().__init__(name, cfg)

