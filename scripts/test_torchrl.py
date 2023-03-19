import functorch
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.learning.collectors import SyncDataCollector
from omni_drones.utils.wandb import init_wandb
from setproctitle import setproctitle

from tensordict.nn import make_functional, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs.utils import step_mdp

from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.functional import vec_generalized_advantage_estimate

from tqdm import tqdm


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs import Hover

    env = Hover(cfg, headless=cfg.headless)
    actor_net = TensorDictModule(
        nn.Sequential(
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(2 * env.drone.action_spec.shape[-1]),
            NormalParamExtractor(),
        ),
        in_keys=["drone.obs"],
        out_keys=["loc", "scale"],
    )
    policy_module: ProbabilisticActor = ProbabilisticActor(
        module=actor_net,
        in_keys=["loc", "scale"],
        out_keys=["drone.action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    ).to(env.device)
    critic_net = nn.Sequential(
        nn.LazyLinear(128),
        nn.LeakyReLU(),
        nn.LazyLinear(128),
        nn.LeakyReLU(),
        nn.LazyLinear(1),
    )
    value_module = ValueOperator(critic_net, in_keys=["drone.obs"]).to(env.device)
    print(policy_module(env.reset()))
    print(value_module(env.reset()))
    loss_module = ClipPPOLoss(
        actor=policy_module,
        critic=value_module,
    )
    collector = SyncDataCollector(
        env,
        policy_module,
        split_trajs=False,
        frames_per_batch=env.num_envs * 16,
        device=cfg.sim.device,
        return_same_td=False,
    )
    actor_optim = torch.optim.Adam(policy_module.parameters(), 0.005)
    critic_optim = torch.optim.Adam(value_module.parameters(), 0.005)

    episode_stats = []

    def record_and_log_stats(done: torch.Tensor):
        done = done.squeeze(-1)
        episode_stats.append(env._tensordict[done].select("drone.return", "progress"))
        if sum(map(len, episode_stats)) >= cfg.env.num_envs:
            print()
            stats = {}
            for k, v in torch.cat(episode_stats).items():
                v = torch.mean(v).item()
                stats[k] = v
                print(f"train/{k}: {v}")
            run.log(stats)
            episode_stats.clear()

    collector.on_reset(record_and_log_stats)

    pbar = tqdm(collector)
    for i, data in enumerate(pbar):
        step_td = step_mdp(data)
        policy_losses = []
        value_losses = []
        for epoch in range(4):
            with torch.no_grad():
                value_module(data)
                value_module(step_td)
            adv, value_target = vec_generalized_advantage_estimate(
                gamma=0.95,
                lmbda=0.95,
                state_value=data["state_value"].squeeze(-2),
                next_state_value=step_td["state_value"].squeeze(-2),
                reward=data[("reward", "drone.reward")].squeeze(-2),
                done=data["done"],
            )
            data.set("adv", adv)
            data.set("value_target", value_target)

            batch = data.reshape(-1)
            for sample_idx in torch.randperm(batch.shape.numel()).reshape(4, -1):
                minibatch = batch[sample_idx]
                dist = policy_module.get_dist(minibatch)
                log_prob = dist.log_prob(minibatch["drone.action"])
                prev_log_prob = minibatch["sample_log_prob"]
                ratio = (log_prob - prev_log_prob).exp()
                policy_loss = -(ratio * minibatch["adv"]).mean()

                state_value = value_module(minibatch).get("state_value").squeeze(-2)
                value_loss = F.smooth_l1_loss(state_value, minibatch["value_target"])

                actor_optim.zero_grad()
                policy_loss.backward()
                actor_optim.step()
                policy_losses.append(policy_loss.detach())

                critic_optim.zero_grad()
                value_loss.backward()
                critic_optim.step()
                value_losses.append(value_loss.detach())

        print("policy loss:", torch.stack(policy_losses).mean().item())
        print("value_loss:", torch.stack(value_losses).mean().item())

        pbar.set_postfix(
            {
                "rollout_fps": collector._fps,
                "frames": collector._frames,
                "episodes": collector._episodes,
            }
        )

    simulation_app.close()


if __name__ == "__main__":
    main()
