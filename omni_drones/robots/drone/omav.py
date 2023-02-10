from omni_drones.robots.robot import RobotBase, ASSET_PATH

class Omav(RobotBase):

    usd_path: str = ASSET_PATH + "/usd/omav.usd"

    def __init__(self, name: str="Omav", cfg=None) -> None:
        super().__init__(name, cfg)

