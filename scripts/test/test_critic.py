from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from tensordict import TensorDict
import torch

from omni_drones.learning.mappo_sp.critic import make_critic

if __name__ == "__main__":
    critic_in_keys = ["observation"]
    critic_out_keys = ["state_value"]

    critic = make_critic(
        hidden_units=[128, 256, 128],
        critic_in_keys=critic_in_keys,
        critic_out_keys=critic_out_keys,
    )

    td = TensorDict(
        {"observation": torch.randn(768, 4, 57)},
        [
            768,
        ],
    )

    output_td = critic(td)
    print(output_td)
