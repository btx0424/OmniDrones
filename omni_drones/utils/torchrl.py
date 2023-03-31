from dataclasses import dataclass
from torchrl.data import TensorSpec
from typing import Optional, Dict
from tensordict.tensordict import TensorDictBase
import time
from torchrl.collectors import SyncDataCollector as _SyncDataCollector

@dataclass
class AgentSpec:
    name: str
    n: int
    observation_spec: TensorSpec
    action_spec: TensorSpec
    reward_spec: TensorSpec
    state_spec: Optional[TensorSpec] = None

class SyncDataCollector(_SyncDataCollector):
    def rollout(self) -> TensorDictBase:
        start = time.perf_counter()
        _tensordict_out = super().rollout()
        self._fps = _tensordict_out.numel() / (time.perf_counter() - start)
        return _tensordict_out