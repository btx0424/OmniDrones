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
    
    def __init__(
        self,
        create_env_fn: Union[
            EnvBase, "EnvCreator", Sequence[Callable[[], EnvBase]]  # noqa: F821
        ],  # noqa: F821
        policy: Optional[
            Union[
                TensorDictModule,
                Callable[[TensorDictBase], TensorDictBase],
            ]
        ] = None,
        total_frames: Optional[int] = -1,
        create_env_kwargs: Optional[dict] = None,
        max_frames_per_traj: int = -1,
        frames_per_batch: int = 200,
        init_random_frames: int = -1,
        reset_at_each_iter: bool = False,
        postproc: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        split_trajs: Optional[bool] = None,
        device: DEVICE_TYPING = None,
        passing_device: DEVICE_TYPING = None,
        seed: Optional[int] = None,
        pin_memory: bool = False,
        exploration_mode: str = DEFAULT_EXPLORATION_MODE,
        init_with_lag: bool = False,
        return_same_td: bool = False,
        reset_when_done: bool = True,
    ):        
        super().__init__(create_env_fn, policy, total_frames, create_env_kwargs, max_frames_per_traj, frames_per_batch, init_random_frames, reset_at_each_iter, postproc, split_trajs, device, passing_device, seed, pin_memory, exploration_mode, init_with_lag, return_same_td, reset_when_done)
        self._reset_callbacks = []
        self._episodes = 0

    def rollout(self) -> TensorDictBase:
        start = time.perf_counter()
        _tensordict_out = super().rollout()
        self._fps = _tensordict_out.numel() / (time.perf_counter() - start)
        return _tensordict_out

    def _reset_if_necessary(self) -> None:
        done: torch.Tensor = self._tensordict.get("done")
        if done.any():
            self._episodes += done.sum().item()
            for callback in self._reset_callbacks:
                callback(done)
        super()._reset_if_necessary()

    def on_reset(self, callback: Callable):
        assert callable(callback)
        self._reset_callbacks.append(callback)

