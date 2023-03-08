import time
import torch
from typing import Union, Optional, Callable, Sequence
from torchrl.collectors.collectors import (
    SyncDataCollector as _SyncDataCollector,
    EnvBase, TensorDictModule, 
    DEFAULT_EXPLORATION_MODE, DEVICE_TYPING
)
from tensordict.tensordict import TensorDictBase

class SyncDataCollector(_SyncDataCollector):

    def rollout(self) -> TensorDictBase:
        start = time.perf_counter()
        _tensordict_out = super().rollout()
        self._fps = _tensordict_out.numel() / (time.perf_counter() - start)
        return _tensordict_out

