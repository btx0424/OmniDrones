from omni_drones.robots.multirotor import Multirotor, RotorCfg
from omni_drones.utils.orbit import DEFAULT_CFG
from omni.isaac.orbit.assets import ArticulationCfg

import os.path as osp

ASSET_PATH = osp.join(osp.dirname(__file__), "assets")

HUMMINGBIRD_CFG = ArticulationCfg(
    class_type=Multirotor,
    spawn=DEFAULT_CFG.replace(usd_path=f"{ASSET_PATH}/usd/hummingbird.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
    ),
    actuators={
        "rotor": RotorCfg(
            body_names_expr=["rotor_.*"],
            max_rotor_speed=838,
            kf=8.54858e-06,
            km=1.3677728816219314e-07,
            rotor_direction={
                "rotor_(0|2)": -1.0,
                "rotor_(1|3)": 1.0,
            },
            tau_up=0.5,
            tau_down=0.5,
        )
    }
)


FIREFLY_CFG = ArticulationCfg(
    class_type=Multirotor,
    spawn=DEFAULT_CFG.replace(usd_path=f"{ASSET_PATH}/usd/firefly.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
    ),
    actuators={
        "rotor": RotorCfg(
            body_names_expr=["rotor_.*"],
            max_rotor_speed=838,
            kf=8.54858e-06,
            km=1.3677728816219314e-07,
            rotor_direction={
                "rotor_(0|2|4)": 1.0,
                "rotor_(1|3|5)": -1.0,
            },
            tau_up=0.5,
            tau_down=0.5,
        )
    }
)