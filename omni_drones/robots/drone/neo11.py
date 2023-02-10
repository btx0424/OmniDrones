from omni_drones.robots.robot import RobotBase, ASSET_PATH

class Neo11(RobotBase):

    usd_path: str = ASSET_PATH + "/usd/neo11.usd"

    def __init__(self, name: str="Neo11", cfg=None) -> None:
        super().__init__(name, cfg)

