from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omegaconf import DictConfig

from omni_drones.views import RigidPrimView
from omni_drones.robots.drone import MultirotorBase
import omni_drones.utils.kit as kit_utils

from omni_drones.utils.torch import euler_to_quaternion
from tensordict.tensordict import TensorDict, TensorDictBase
import omni.isaac.core.objects as objects
import omni.isaac.core.materials as materials

import torch
import torch.distributions as D
import torch.functional as F
import torch.nn.functional as NNF

from pxr import UsdShade
from typing import List

from torchrl.data import (
    UnboundedContinuousTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
)


class PingPong2v2(IsaacEnv):
    def __init__(self, cfg: DictConfig, headless: bool):
        super().__init__(cfg, headless)

        self.time_encoding: bool = self.cfg.task.get("time_encoding", False)
        
        self.drone.initialize()
        
        self.ball = RigidPrimView(
            "/World/envs/env_*/ball",
            reset_xform_properties=False,
            track_contact_forces=True,
            shape=(-1, 1),
        )

        self.ball.initialize()

        self.init_ball_offset_dist = D.Uniform(
            torch.tensor([-0.3, -0.3, 1.6], device=self.device),
            torch.tensor([0.3, 0.3, 2.0], device=self.device),
        )

        z_low, z_high = 1.5, 1.7
        x_low, x_high = 0.5, 1.7
        y_low, y_high = 0.5, 1.7

        self.init_drone_pos_dist = D.Uniform(
            torch.tensor(
                [
                    [-x_high, -y_high, z_low],
                    [-x_high, y_low, z_low],
                    [x_low, y_low, z_low],
                    [x_low, -y_high, z_low],
                ],
                device=self.device,
            ),
            torch.tensor(
                [
                    [-x_low, -y_low, z_high],
                    [-x_low, y_high, z_high],
                    [x_high, y_high, z_high],
                    [x_high, -y_low, z_high],
                ],
                device=self.device,
            ),
        )
        self.init_drone_rpy_dist = D.Uniform(
            torch.tensor([-0.1, -0.1, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, 2.0], device=self.device) * torch.pi,
        )

        # turn[env_id] \in {0,1,2,3}. drone turn[env_id] should relay the ball to (turn[env_id]+1)%4
        self.turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)

        self.last_ball_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)
