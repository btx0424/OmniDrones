import time
from typing import Callable, Optional, Sequence, Union

import torch
from tensordict.tensordict import TensorDictBase
from torchrl.collectors.collectors import (
    DEFAULT_EXPLORATION_MODE,
    DEVICE_TYPING,
    EnvBase,
    SyncDataCollector as _SyncDataCollector,
    TensorDictModule,
)


class SyncDataCollector(_SyncDataCollector):
    def rollout(self) -> TensorDictBase:
        start = time.perf_counter()
        _tensordict_out = super().rollout()
        self._fps = _tensordict_out.numel() / (time.perf_counter() - start)
        return _tensordict_out
