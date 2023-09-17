import logging
import os
import time

import hydra
import torch
import numpy as np
import wandb
from functorch import vmap
from omegaconf import OmegaConf, DictConfig

from omni.isaac.kit import SimulationApp
from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.torchrl import SyncDataCollector, AgentSpec
from omni_drones.utils.wandb import init_wandb

from setproctitle import setproctitle
from torchrl.envs.transforms import (
    TransformedEnv,
    InitTracker,
    Compose,
)

from tqdm import tqdm
from typing import Callable, Dict
import traceback


class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1


def get_policy(cfg: DictConfig, agent_spec: AgentSpec):
    from omni_drones.learning import MAPPOSPPolicy

    algos = {
        "mappo_sp": MAPPOSPPolicy,
    }
    algo_name = cfg.algo.name.lower()
    if algo_name not in algos:
        raise RuntimeError(f"{algo_name} not supported.")
    policy = algos[algo_name](cfg.algo, agent_spec=agent_spec, device="cuda")
    return policy


def get_transforms(
    cfg: DictConfig,
    base_env,
    logger_func: Callable[
        [
            Dict,
        ],
        None,
    ],
):
    from omni_drones.utils.torchrl.transforms import (
        LogOnEpisode,
        FromMultiDiscreteAction,
        FromDiscreteAction,
        ravel_composite,
        History,
    )

    stats_keys = [
        k
        for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    logger = LogOnEpisode(
        cfg.env.num_envs,
        in_keys=stats_keys,
        log_keys=stats_keys,
        logger_func=logger_func,
        process_func=None,
    )
    transforms = [InitTracker(), logger]

    # a CompositeSpec is by deafault processed by a entity-based encoder
    # flatten it to use a MLP encoder instead
    if cfg.task.get("flatten_obs", False):
        transforms.append(
            ravel_composite(base_env.observation_spec, ("agents", "observation"))
        )
    if cfg.task.get("flatten_state", False):
        transforms.append(ravel_composite(base_env.observation_spec, "state"))
    if cfg.task.get("flatten_intrinsics", True) and (
        "agents",
        "intrinsics",
    ) in base_env.observation_spec.keys(True):
        transforms.append(
            ravel_composite(
                base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1
            )
        )

    if cfg.task.get("history", False):
        transforms.append(History([("agents", "observation")]))

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform == "velocity":
            from omni_drones.controllers import LeePositionController
            from omni_drones.utils.torchrl.transforms import VelController

            controller = LeePositionController(9.81, base_env.drone.params).to(
                base_env.device
            )
            transform = VelController(controller)
            transforms.append(transform)
        elif action_transform == "attitude":
            from omni_drones.controllers import AttitudeController as Controller
            from omni_drones.utils.torchrl.transforms import AttitudeController

            controller = Controller(9.81, base_env.drone.params).to(base_env.device)
            transform = AttitudeController(controller)
            transforms.append(transform)
        elif action_transform == "rate":
            from omni_drones.controllers import RateController as _RateController
            from omni_drones.utils.torchrl.transforms import RateController

            controller = _RateController(9.81, base_env.drone.params).to(
                base_env.device
            )
            transform = RateController(controller)
            transforms.append(transform)
        elif not action_transform.lower() == "none":
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    return transforms


def train(
    cfg: DictConfig, simulation_app: SimulationApp, env: TransformedEnv, wandb_run
):
    # agent_spec: AgentSpec = env.agent_spec["drone"]
    # policy = get_policy(cfg, agent_spec)
    # if cfg.get("resume_ckpt_path") is not None:
    #     policy.load_state_dict(torch.load(cfg.resume_ckpt_path))
    #     print(f"resume policy from {cfg.resume_ckpt_path}")

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters:int = cfg.get("max_iters", -1)
    eval_interval:int = cfg.get("eval_interval", -1)
    save_interval:int = cfg.get("save_interval", -1)

    collector = SyncDataCollector(
        env,
        policy=None,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    pbar = tqdm(collector)
    env.train()
    for i, data in enumerate(pbar):
        pass


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_sp")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    def log(info):
        # print(OmegaConf.to_yaml(info))
        run.log(info)

    transforms = get_transforms(cfg, base_env, log)

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    try:
        train(cfg=cfg, simulation_app=simulation_app, env=env, wandb_run=run)
    except Exception as e:
        traceback.print_exc()

    finally:
        wandb.save(os.path.join(run.dir, "checkpoint*"))
        wandb.finish()

        simulation_app.close()


if __name__ == "__main__":
    main()
