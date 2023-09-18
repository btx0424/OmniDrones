import torch
import time
from torchrl.collectors import SyncDataCollector as _SyncDataCollector
from torchrl.collectors.utils import split_trajectories
from tensordict.tensordict import TensorDictBase

from typing import Iterator

class SyncDataCollector(_SyncDataCollector):
    def rollout(self) -> TensorDictBase:
        start = time.perf_counter()
        _tensordict_out = super().rollout()
        self._fps = _tensordict_out.numel() / (time.perf_counter() - start)
        return _tensordict_out
    
    def iterator(self) -> Iterator[TensorDictBase]:
        """Iterates through the DataCollector.

        Yields: TensorDictBase objects containing (chunks of) trajectories

        """
        
        total_frames = self.total_frames
        i = -1
        self._frames = 0
        while True:
            i += 1
            self._iter = i
            tensordict_out = self.rollout()
            self._frames += tensordict_out.numel()
            if self._frames >= total_frames:
                self.env.close()

            if self.split_trajs:
                tensordict_out = split_trajectories(
                    tensordict_out, prefix="collector"
                )
            if self.postproc is not None:
                tensordict_out = self.postproc(tensordict_out)
            if self._exclude_private_keys:
                excluded_keys = [
                    key for key in tensordict_out.keys() if key.startswith("_")
                ]
                tensordict_out = tensordict_out.exclude(
                    *excluded_keys, inplace=True
                )
            if self.return_same_td:
                yield tensordict_out
            else:
                # we must clone the values, as the tensordict is updated in-place.
                # otherwise the following code may break:
                # >>> for i, data in enumerate(collector):
                # >>>      if i == 0:
                # >>>          data0 = data
                # >>>      elif i == 1:
                # >>>          data1 = data
                # >>>      else:
                # >>>          break
                # >>> assert data0["done"] is not data1["done"]
                yield tensordict_out.clone()

            if self._frames >= self.total_frames:
                break