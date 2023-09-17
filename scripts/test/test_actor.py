from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from tensordict import TensorDict
import torch

from omni_drones.learning.mappo_sp.actor import make_actor


if __name__ == "__main__":
    actor_in_keys = ["observation"]
    actor_out_keys = [("agent", "action")]
    actor_spec = UnboundedContinuousTensorSpec(
        shape=(4,),
    )
    actor = make_actor(
        hidden_units=[128, 256, 128],
        actor_in_keys=actor_in_keys,
        actor_out_keys=actor_out_keys,
        action_spec=actor_spec,
    )
    td = TensorDict(
        {"observation": torch.randn(768, 4, 57)},
        [
            768,
        ],
    )

    output_td = actor(td)
    print(output_td)
