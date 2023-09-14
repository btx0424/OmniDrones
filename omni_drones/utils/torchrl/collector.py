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
    
    def _step_and_maybe_reset(self) -> None:

        any_done = False
        done_map = {}
        for done_key in self.env.done_keys:
            done = self._tensordict.get(("next", done_key))
            truncated = self._tensordict.get(
                ("next", _replace_last(done_key, "truncated")),
                None,
            )
            done = (done | truncated) if truncated is not None else done
            any_sub_done = done.any().item()
            if any_sub_done and self.reset_when_done:
                # Add this done to the map, we will need it to reset
                done_map.update({done_key: done.clone()})
            any_done += any_sub_done

        self._tensordict = step_mdp(
            self._tensordict,
            reward_keys=self.env.reward_keys,
            done_keys=self.env.done_keys,
            action_keys=self.env.action_keys,
        )

        if not self.reset_when_done:
            return

        if any_done:
            traj_ids = self._tensordict.get(("collector", "traj_ids"))
            traj_ids = traj_ids.clone()
            # collectors do not support passing other tensors than `"_reset"`
            # to `reset()`.
            td_reset = self._tensordict.select(*done_map.keys())
            for done_key, done in done_map.items():
                td_reset.set(_replace_last(done_key, "_reset"), done)
                del td_reset[done_key]
            td_reset = self.env.reset(td_reset)
            for done_key in done_map.keys():
                del td_reset[_replace_last(done_key, "_reset")]

            traj_done_or_terminated = torch.stack(
                [
                    done.sum(
                        tuple(range(self._tensordict.batch_dims, done.ndim)),
                        dtype=torch.bool,
                    )
                    for done in done_map.values()
                ],
                dim=0,
            ).any(0)

            if td_reset.batch_dims:
                # better cloning here than when passing the td for stacking
                # cloning is necessary to avoid modifying entries in-place
                self._tensordict = self._tensordict.clone()
                self._tensordict[traj_done_or_terminated] = td_reset[traj_done_or_terminated]
                # self._tensordict = torch.where(
                #     traj_done_or_terminated, td_reset, self._tensordict
                # )
            else:
                self._tensordict.update(td_reset)

            traj_ids[traj_done_or_terminated] = traj_ids.max() + torch.arange(
                1, traj_done_or_terminated.sum() + 1, device=traj_ids.device
            )
            self._tensordict.set(("collector", "traj_ids"), traj_ids)

