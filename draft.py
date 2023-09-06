from omni_drones.learning.mappo import MAPPOPolicy
from omni_drones import CONFIG_PATH
import hydra
from omni_drones.utils.torchrl.env import AgentSpec, DummyEnv
import torch
from omegaconf import OmegaConf

from torchrl.data import (
    UnboundedContinuousTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    BoundedTensorSpec,
)


def get_agent_spec():
    observation_dim = 61
    state_dim = 127

    num_envs = 16
    device = "cuda"

    observation_spec = (
        CompositeSpec(
            {
                "agents": {
                    "observation": UnboundedContinuousTensorSpec((4, observation_dim)),
                    "state": UnboundedContinuousTensorSpec(state_dim),
                }
            }
        )
        .expand(num_envs)
        .to(device)
    )

    drone_num_rotors = 4
    drone_action_spec = BoundedTensorSpec(-1, 1, drone_num_rotors, device=device)
    action_spec = (
        CompositeSpec(
            {
                "agents": {
                    "action": torch.stack([drone_action_spec] * 4, dim=0),
                }
            }
        )
        .expand(num_envs)
        .to(device)
    )

    reward_spec = (
        CompositeSpec({"agents": {"reward": UnboundedContinuousTensorSpec((4, 1))}})
        .expand(num_envs)
        .to(device)
    )
    dummy_env = DummyEnv(
        observation_spec=observation_spec,
        action_spec=action_spec,
        reward_spec=reward_spec,
    )

    agent_spec = AgentSpec(
        "drone",
        4,
        observation_key=("agents", "observation"),
        action_key=("agents", "action"),
        reward_key=("agents", "reward"),
        state_key=("agents", "state"),
        _env=dummy_env,
    )

    return agent_spec


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="eval")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    agent_spec = get_agent_spec()

    policy = MAPPOPolicy(cfg=cfg.algo, agent_spec=agent_spec)

    state_dict = torch.load(cfg.policy_ckpt_path)
    policy.load_state_dict(state_dict)


if __name__ == "__main__":
    main()
