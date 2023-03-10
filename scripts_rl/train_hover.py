import torch
import hydra
import os
import time

from tqdm import tqdm
from omegaconf import OmegaConf
from setproctitle import setproctitle
from functorch import vmap
from tensordict import TensorDict
from torchrl.data import UnboundedContinuousTensorSpec

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.learning.collectors import SyncDataCollector
from omni_drones.utils.wandb import init_wandb

class Every:
    def __init__(self, func, k=1) -> None:
        self.func = func
        self.k = k
        self.count = 0
    
    def __call__(self, *args, **kwds):
        if self.count % self.k == 0:
            self.func(*args, **kwds)
        self.count = (self.count + 1) % self.k

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs import Hover
    from omni_drones.learning.mappo import MAPPOPolicy
    from omni_drones.sensors.camera import Camera

    env = Hover(cfg, headless=cfg.headless)
    agent_spec = env.agent_spec["drone"]
    agent_spec.action_spec = UnboundedContinuousTensorSpec(3, device=env.device)
    ppo = MAPPOPolicy(cfg.algo, agent_spec, act_name="drone.target_vel", device="cuda")
    controller = env.drone.default_controller(
        env.drone.dt, 9.81, env.drone.params
    ).to(env.device)

    def policy(tensordict: TensorDict):
        tensordict = ppo(tensordict)
        relative_state = tensordict["drone.obs"][..., :13].clone()
        target_vel = tensordict["drone.target_vel"]
        control_target = torch.cat([
            relative_state[..., :3], target_vel, torch.zeros_like(target_vel[..., [0]])], dim=-1)
        controller_state = tensordict.get("controller_state", TensorDict({}, relative_state.shape[:2]))
                
        cmds, controller_state = vmap(vmap(controller))(relative_state, control_target, controller_state)
        torch.nan_to_num_(cmds, 0.)
        assert not torch.isnan(cmds).any()
        tensordict["drone.action"] = cmds #command for motor
        tensordict["controller_state"] = controller_state 
        return tensordict
    
    collector = SyncDataCollector(
        env, 
        policy, 
        split_trajs=False,
        frames_per_batch=env.num_envs * cfg.algo.train_every,
        device=cfg.sim.device, 
        return_same_td=True,
    )

    episode_stats = []
    def record_and_log_stats(done: torch.Tensor):
        done = done.squeeze(-1)
        episode_stats.append(env._tensordict[done])
        if sum(map(len, episode_stats)) >= 4096:
            stats = {}
            for k, v in torch.cat(episode_stats).items():
                v = torch.mean(v).item()
                stats[k] = v
                print(f"train/{k}: {v}")
            run.log(stats)
            episode_stats.clear()
    collector.on_reset(record_and_log_stats)
    logger = Every(lambda info: run.log(info), 10)

    frames = []
    camera = Camera(translation=(1., 1., 1.), target=(0., 0., 0.5))
    def record_video(env, tensordict):
        frames.append(camera().clone())
    def eval_op():
        eval_rollout = env.rollout(
            policy,
            max_steps=500,
            callback=record_video,
            break_when_any_done=False
        )
        videos = torch.stack(frames)
    evaluator = Every(eval_op, 100)

    pbar = tqdm(collector)
    for i, data in enumerate(pbar):
        info = ppo.train_op(data.clone())
        logger(info)
        evaluator()
        pbar.set_postfix({
            "rollout_fps": collector._fps,
            "frames": collector._frames,
            "episodes": collector._episodes,
        })
        
    simulation_app.close()

if __name__ == "__main__":
    main()