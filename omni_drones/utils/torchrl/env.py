from dataclasses import dataclass
from torchrl.envs import EnvBase
from torchrl.data import TensorSpec, CompositeSpec
from torchrl.collectors import SyncDataCollector as _SyncDataCollector
from tensordict.tensordict import TensorDictBase

from typing import Optional
import time

@dataclass
class AgentSpec:
    name: str
    n: int
    observation_key: Optional[str] = "observation"
    action_key: Optional[str] = None
    state_key: Optional[str] = None
    reward_key: Optional[str] = None
    done_key: Optional[str] = None

    _env: Optional[EnvBase] = None

    @property
    def observation_spec(self) -> TensorSpec:
        return self._env.observation_spec[self.observation_key]

    @property
    def action_spec(self) -> TensorSpec:
        if self.action_key is None:
            return self._env.action_spec
        try:
            return self._env.input_spec["_action_spec"][self.action_key]
        except:
            return self._env.action_spec[self.action_key]
    
    @property
    def state_spec(self) -> TensorSpec:
        if self.state_key is None:
            if isinstance(self._env.state_spec, CompositeSpec):
                if len(self._env.state_spec) == 0:
                    return None
            return self._env.state_spec
        return self._env.state_spec[self.state_key]
    
    @property
    def reward_spec(self) -> TensorSpec:
        if self.reward_key is None:
            return self._env.reward_spec
        try:
            return self._env.output_spec["_reward_spec"][self.reward_key]
        except:
            return self._env.reward_spec[self.reward_key]

    @property
    def done_spec(self) -> TensorSpec:
        if self.done_key is None:
            return self._env.done_spec
        try:
            return self._env.output_spec["_done_spec"][self.done_key]
        except:
            return self._env.done_spec[self.done_key]


class SyncDataCollector(_SyncDataCollector):
    def rollout(self) -> TensorDictBase:
        start = time.perf_counter()
        _tensordict_out = super().rollout()
        self._fps = _tensordict_out.numel() / (time.perf_counter() - start)
        return _tensordict_out