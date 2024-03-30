import torch
from omni_drones.robots.multirotor import Multirotor

from .mdp_term import MDPTerm


class ObservationFunc(MDPTerm):
    pass


class DynamicState(ObservationFunc):

    def __init__(self, env: "IsaacEnv", asset_name: str="drone"):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]

    def compute(self) -> torch.Tensor:
        state = torch.cat([
            self.asset.data.root_pos_w[..., [2]],
            self.asset.data.root_quat_w,
            self.asset.data.root_vel_w,
            self.asset.data.projected_gravity_b,
            # self.asset.data.heading,
            *self.asset.data.throttle.values()
        ], dim=-1)
        return state.reshape(*self.asset.shape, -1)


class BodyDynamicState(ObservationFunc):
    
    def __init__(
        self, 
        env: "IsaacEnv", 
        asset_name: str="drone",
        body_name: str="payload"
    ):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]
        self.body_id = self.asset.find_bodies(body_name)[0]
    
    def compute(self) -> torch.Tensor:
        body_pos = (
            self.asset.data.body_pos_w[..., self.body_id, :]
            - self.asset.data.root_pos_w.unsqueeze(-2)
        )
        state = torch.cat([
            body_pos,
            self.asset.data.body_quat_w[..., self.body_id, :],
            self.asset.data.body_vel_w[..., self.body_id, :],
        ], dim=-1)
        return state.reshape(*self.asset.shape, -1)