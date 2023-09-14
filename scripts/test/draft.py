from omni_drones import CONFIG_PATH

from omegaconf import DictConfig, OmegaConf

import hydra

from typing import Dict
import torch
from torch import Tensor
import torch.distributions as D


def get_extended_pos_dist(
    x_low: float,
    y_low: float,
    z_low: float,
    x_high: float,
    y_high: float,
    z_high: float,
    device,
):
    return D.Uniform(
        torch.tensor(
            [
                [-x_high, -y_high, z_low],
                [-x_high, y_low, z_low],
                [x_low, y_low, z_low],
                [x_low, -y_high, z_low],
            ],
            device=device,
        ),
        torch.tensor(
            [
                [-x_low, -y_low, z_high],
                [-x_low, y_high, z_high],
                [x_high, y_high, z_high],
                [x_high, -y_low, z_high],
            ],
            device=device,
        ),
    )



@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_sp")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    device = "cpu"

    task_cfg: DictConfig = cfg.task
    init_drone_pos_dist = get_extended_pos_dist(
            *task_cfg.initial.drone_xyz_dist.low,
            *task_cfg.initial.drone_xyz_dist.high,
            device=device
        )
    print(init_drone_pos_dist)
    a=torch.zeros(256, device=device, dtype=torch.bool)
    print(a.shape)
    print(a[:5])


if __name__ == "__main__":
    main()
