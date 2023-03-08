import torch
from torchrl.envs.transforms import Transform
from tensordict.tensordict import TensorDictBase
from typing import Sequence, Any, Dict
from collections import defaultdict, Callable


class LogOnEpisode(Transform):
    def __init__(
        self, 
        n_episodes: int,
        in_keys: Sequence[str]=None,
        log_keys: Sequence[str]=None,
        logger_func: Callable=None,
        process_func: Dict[str, Callable]=None
    ):
        super().__init__(in_keys=in_keys)
        if not len(in_keys) == len(log_keys):
            raise ValueError
        
        self.log_keys = log_keys
        self.n_episodes = n_episodes
        self.logger_func = logger_func
        self.process_func = defaultdict(lambda: lambda x: torch.mean(x).item())
        if process_func is not None:
            self.process_func.update(process_func)

        self.stats = []

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        done = tensordict.get(
            "done",
            torch.zeros(
                tensordict.batch_size,
                dtype=torch.bool,
                device=tensordict.device,
            )
        ).squeeze(-1)
        if done is not None and done.any():
            self.stats.extend(
                tensordict[done]
                .select(*self.in_keys)
                .clone()
                .unbind(0)
            )
            if len(self.stats) >= self.n_episodes:
                stats: TensorDictBase = torch.stack(self.stats)
                dict_to_log = {}
                for in_key, log_key in zip(self.in_keys, self.log_keys):
                    if in_key in stats.keys():
                        process_func = self.process_func[in_key]
                        dict_to_log[log_key] = process_func(stats[in_key])
                if self.logger_func is not None:
                    self.logger_func(dict_to_log)
                self.stats.clear()
        return tensordict
        
        