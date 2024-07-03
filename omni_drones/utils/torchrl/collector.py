# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import time
from torchrl.collectors import SyncDataCollector as _SyncDataCollector
from torchrl.collectors.utils import split_trajectories
from torchrl.envs.utils import _replace_last, step_mdp
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
            # if self._frames >= total_frames:
            #     self.env.close()

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

