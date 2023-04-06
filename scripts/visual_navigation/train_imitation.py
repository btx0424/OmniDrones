import logging
import os
import time

import hydra
import torch
import torch.nn as nn
import wandb
from functorch import vmap
from omegaconf import OmegaConf
from tqdm import tqdm
from setproctitle import setproctitle

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import (
    TensorSpec,
    UnboundedContinuousTensorSpec,
    CompositeSpec,
    TensorDictReplayBuffer,
    LazyMemmapStorage
)

from torchrl.envs.transforms import (
    TransformedEnv, 
    InitTracker, 
    Compose,
    CatTensors
)
from torchvision.models import mobilenet_v3_small
from torchvision.io import write_video

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.envs.transforms import LogOnEpisode, DepthImageNorm
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.math import quaternion_to_euler
from omni_drones.learning.utils.distributions import IndependentNormal
from omni_drones.learning.utils.network import MLP
from omni_drones.utils.torchrl import SyncDataCollector


def make_dataset(
    traj_dir: str,
    batch_size: int,
    max_size: 32000,
):
    buffer = TensorDictReplayBuffer(
        batch_size=batch_size,
        storage=LazyMemmapStorage(max_size, scratch_dir="tmp"),
    )
    for traj_file in tqdm(os.listdir(traj_dir)[:10], desc="loading data"):
        td: TensorDict = torch.load(os.path.join(traj_dir, traj_file))
        buffer.extend(td.reshape(-1))
    return buffer

class NormalParam(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.linear = nn.LazyLinear(output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, input):
        loc = self.linear(input)
        scale = torch.broadcast_to(torch.exp(self.log_std), loc.shape)
        return loc, scale
    
class Actor(nn.Module):

    def forward(self, loc, scale):
        dist = IndependentNormal(loc, scale)
        action = dist.sample()
        logp = dist.log_prob(action).unsqueeze(-1)
        return action, logp

class ActorEval(nn.Module):

    def forward(self, action, loc, scale):
        dist = IndependentNormal(loc, scale)
        logp = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy()
        return logp, entropy

def make_model(
    input_spec: CompositeSpec,
    action_spec: TensorSpec,
    device: torch.device="cpu"
):
    visual_encoder = nn.Sequential(
        nn.Conv2d(1, 3, 1),
        mobilenet_v3_small(num_classes=128)
    )
    state_input_shape = input_spec["state"].shape
    state_encoder = MLP([state_input_shape[-1], 128], normalization=nn.LayerNorm)
    encoder = TensorDictSequential(
        TensorDictModule(visual_encoder, [("drone.obs", "distance_to_camera")], ["visual_feature"]),
        TensorDictModule(state_encoder, [("drone.obs", "state")], ["state_feature"]),
        CatTensors(["state_feature", "visual_feature"], "feature"),
        TensorDictModule(MLP([128 + 128, 128]), ["feature"], ["feature"]),
        TensorDictModule(NormalParam(action_spec.shape[-1]), ["feature"], ["loc", "scale"]),
    ).to(device)
    actor = TensorDictSequential(
        encoder,
        TensorDictModule(Actor(), ["loc", "scale"], ["action", "logp"])
    )
    actor_eval = TensorDictSequential(
        encoder,
        TensorDictModule(ActorEval(), ["control_target", "loc", "scale"], ["logp", "entropy"])
    )
    return actor, actor_eval 

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs import Forest
    from omni_drones.learning.mappo import MAPPOPolicy

    base_env = Forest(cfg, headless=cfg.headless)

    actor, actor_eval = make_model(
        input_spec=CompositeSpec({
            "state": UnboundedContinuousTensorSpec(25),
        }),
        action_spec=UnboundedContinuousTensorSpec(6),
        device=base_env.device
    )

    controller = base_env.drone.DEFAULT_CONTROLLER(base_env.drone.dt, 9.81, base_env.drone.params).to(
        base_env.device
    )
    
    def policy(tensordict: TensorDict):
        actor_input = tensordict.select(*actor.in_keys)
        actor_input.batch_size = [*tensordict.batch_size, 1]
        actor_input = actor_input.squeeze(1)
        actor_output = actor(actor_input).unsqueeze(1)

        drone_state = tensordict[("info", "drone_state")]
        target_pos_vel = actor_output["control_target"]
        target_yaw = torch.zeros_like(target_pos_vel[..., 0].unsqueeze(-1))

        control_target = torch.cat([target_pos_vel, target_yaw], dim=-1)

        controller_state = tensordict.get(
            "controller_state", TensorDict({}, drone_state.shape[:2])
        )
        cmds, controller_state = vmap(vmap(controller))(
            drone_state, control_target, controller_state
        )
        torch.nan_to_num_(cmds, 0.0)
        assert not torch.isnan(cmds).any()
        tensordict["drone.action"] = cmds
        tensordict["controller_state"] = controller_state
        return tensordict

    def log(info):
        print(OmegaConf.to_yaml(info))
        run.log(info)

    logger = LogOnEpisode(
        cfg.env.num_envs,
        in_keys=["return", "progress"],
        log_keys=["return", "ep_length"],
        logger_func=log,
    )

    transforms = Compose(
        InitTracker(), 
        DepthImageNorm([("drone.obs", "distance_to_camera")], 0, 24),
        logger
    )
    env = TransformedEnv(base_env, transforms)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=base_env.num_envs * base_env.max_episode_length,
        total_frames=-1,
        device=cfg.sim.device,
        storing_device="cpu",
    )
    
    opt = torch.optim.Adam(actor.parameters())
    buffer = make_dataset("trajectories", 128, 32000)
    for epoch in range(5):
        t = tqdm(range(512))
        actor.train()
        for _ in t:
            batch = buffer.sample().to(env.device)
            batch["drone.obs"] = batch["drone.obs"].squeeze(1)
            batch["drone.action"] = batch["drone.action"].squeeze(1)
            actor_output = actor_eval(batch)
            loss = - torch.mean(actor_output["logp"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            t.set_postfix({"loss": loss.item()})
        actor.eval()
        
        with torch.no_grad():
            td = next(collector)

    simulation_app.close()


if __name__ == "__main__":
    main()
