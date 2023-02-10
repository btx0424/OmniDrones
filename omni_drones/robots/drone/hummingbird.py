from omni_drones.robots.robot import RobotBase, ASSET_PATH

class Hummingbird(RobotBase):

    usd_path: str = ASSET_PATH + "/usd/hummingbird.usd"

    def __init__(self, name: str="Hummingbird", cfg=None) -> None:
        super().__init__(name, cfg)

