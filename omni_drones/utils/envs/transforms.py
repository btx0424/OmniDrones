from collections import Callable, defaultdict
from typing import Any, Dict, Optional, Sequence, Union

import torch
from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.transforms import (
    Transform,
    Compose,
    FlattenObservation,
    CatTensors
)
from torchrl.data import (
    TensorSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    CompositeSpec,
)


class LogOnEpisode(Transform):
    def __init__(
        self,
        n_episodes: int,
        in_keys: Sequence[str] = None,
        log_keys: Sequence[str] = None,
        logger_func: Callable = None,
        process_func: Dict[str, Callable] = None,
    ):
        super().__init__(in_keys=in_keys)
        if not len(in_keys) == len(log_keys):
            raise ValueError
        self.in_keys = in_keys
        self.log_keys = log_keys

        self.n_episodes = n_episodes
        self.logger_func = logger_func
        self.process_func = defaultdict(lambda: lambda x: torch.mean(x.float()).item())
        if process_func is not None:
            self.process_func.update(process_func)

        self.stats = []
        self._frames = 0

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        _reset = tensordict.get(("next", "done"), None)
        if _reset is None:
            _reset = torch.ones(
                tensordict.batch_size, dtype=torch.bool, device=tensordict.device
            )
        if _reset.any():
            _reset = _reset.all(-1).nonzero().squeeze(-1)
            self.stats.extend(
                tensordict[_reset].select(*self.in_keys).clone().unbind(0)
            )
            if len(self.stats) >= self.n_episodes:
                stats: TensorDictBase = torch.stack(self.stats)
                dict_to_log = {}
                for in_key, log_key in zip(self.in_keys, self.log_keys):
                    if in_key in stats.keys(True, True):
                        process_func = self.process_func[in_key]
                        if isinstance(log_key, tuple):
                            log_key = ".".join(log_key)
                        dict_to_log[log_key] = process_func(stats[in_key])
                
                if self.training:
                    dict_to_log = {f"train/{k}": v for k, v in dict_to_log.items()}
                else:
                    dict_to_log = {f"eval/{k}": v for k, v in dict_to_log.items()}
                
                if self.logger_func is not None:
                    dict_to_log["env_frames"] = self._frames
                    self.logger_func(dict_to_log)
                self.stats.clear()
        
        if self.training:
            self._frames += tensordict.numel()
        return tensordict


class FromDiscreteAction(Transform):
    def __init__(
        self,
        action_key: Sequence[str] = None,
        nbins: Union[int, Sequence[int]] = None,
    ):
        if action_key is None:
            action_key = "action"
        if nbins is None:
            nbins = 2
        super().__init__([], in_keys_inv=[action_key])
        self.nbins = nbins
        self.action_key = action_key

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        action_spec = input_spec[self.action_key]
        if isinstance(action_spec, BoundedTensorSpec):
            if isinstance(self.nbins, int):
                nbins = [self.nbins] * action_spec.shape[-1]
            elif len(self.nbins) == action_spec.shape[-1]:
                nbins = self.nbins
            else:
                raise ValueError(
                    "nbins must be int or list of length equal to the last dimension of action space."
                )
            self.minimum = action_spec.space.minimum.unsqueeze(-2)
            self.maximum = action_spec.space.maximum.unsqueeze(-2)
            self.mapping = torch.cartesian_prod(
                *[torch.linspace(0, 1, dim_nbins) for dim_nbins in nbins]
            ).to(action_spec.device)  # [prod(nbins), len(nbins)]
            n = self.mapping.shape[0]
            spec = DiscreteTensorSpec(
                n, shape=[*action_spec.shape[:-1], 1], device=action_spec.device
            )
            input_spec[self.action_key] = spec
        else:
            NotImplementedError("Only BoundedTensorSpec is supported.")
        return input_spec

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        mapping = self.mapping * (self.maximum - self.minimum) + self.minimum
        action = action.unsqueeze(-1)
        action = torch.take_along_dim(mapping, action, dim=-2).squeeze(-2)
        return action


class FromMultiDiscreteAction(Transform):
    def __init__(
        self,
        action_key: Sequence[str] = None,
        nbins: Union[int, Sequence[int]] = 2,
    ):
        if action_key is None:
            action_key = "action"
        super().__init__([], in_keys_inv=[action_key])
        self.nbins = nbins
        self.action_key = action_key

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        action_spec = input_spec[self.action_key]
        if isinstance(action_spec, BoundedTensorSpec):
            if isinstance(self.nbins, int):
                nbins = [self.nbins] * action_spec.shape[-1]
            elif len(self.nbins) == action_spec.shape[-1]:
                nbins = self.nbins
            else:
                raise ValueError(
                    "nbins must be int or list of length equal to the last dimension of action space."
                )
            spec = MultiDiscreteTensorSpec(
                nbins, shape=action_spec.shape, device=action_spec.device
            )
            self.nvec = spec.nvec.to(action_spec.device)
            self.minimum = action_spec.space.minimum
            self.maximum = action_spec.space.maximum
            input_spec[self.action_key] = spec
        else:
            NotImplementedError("Only BoundedTensorSpec is supported.")
        return input_spec

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        action = action / (self.nvec - 1) * (self.maximum - self.minimum) + self.minimum
        return action

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return super()._inv_call(tensordict)


class DepthImageNorm(Transform):
    def __init__(
        self,
        in_keys: Sequence[str],
        min_range: float,
        max_range: float,
        inverse: bool=False
    ):
        super().__init__(in_keys=in_keys)
        self.max_range = max_range
        self.min_range = min_range
        self.inverse = inverse

    def _apply_transform(self, obs: torch.Tensor) -> None:
        obs = torch.nan_to_num(obs, posinf=self.max_range, neginf=self.min_range)
        obs = obs.clip(self.min_range, self.max_range)
        if self.inverse:
            obs = (obs - self.min_range) / (self.max_range - self.min_range)
        else:
            obs = (self.max_range - obs) / (self.max_range - self.min_range)
        return obs


def flatten_composite(spec: CompositeSpec, key: str):
    composite_spec = spec[key]
    if isinstance(composite_spec, CompositeSpec):
        in_keys = [k for k in spec.keys(True, True) if k[0] == key]
        return Compose(
            FlattenObservation(-2, -1, in_keys),
            CatTensors(in_keys, out_key=key)
        )
    else:
        raise TypeError


class VelController(Transform):
    def __init__(
        self,
        controller,
        action_key: str,
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state"), "controller_state"])
        self.controller = controller
        self.action_key = action_key
    
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[self.action_key]
        spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+(3,), device=action_spec.device)
        input_spec[self.action_key] = spec
        return input_spec
    
    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        shape = tensordict[("info", "drone_state")].shape
        tensordict.set("controller_state", TensorDict({}, shape[:-1]))
        return tensordict
    
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        controller_state = tensordict["controller_state"]
        target_vel = tensordict[("action", "drone.action")]
        control_target = torch.cat([
            drone_state[..., :3],
            target_vel,
            torch.zeros_like(target_vel[..., [0]]),
        ], dim=-1)
        cmds, controller_state = self.controller(drone_state, control_target, controller_state)
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(("action", "drone.action"), cmds)
        tensordict.set("controller_state", controller_state)
        return tensordict


from collections import defaultdict
class History(Transform):
    def __init__(
        self,
        in_keys: Sequence[str],
        out_keys: Sequence[str]=None,
        steps: int = 5,
    ):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.steps = steps
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        history_dict = {}
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            is_tuple = isinstance(in_key, tuple)
            if in_key in observation_spec.keys(include_nested=is_tuple):
                spec = observation_spec[in_key]
                spec = spec.unsqueeze(spec.ndim-1).expand(*spec.shape[:-1], self.steps, spec.shape[-1])
                observation_spec[out_key] = spec
            elif in_key in self.parent.input_spec.keys(include_nested=is_tuple):
                spec = self.parent.input_spec[in_key]
                spec = spec.unsqueeze(spec.ndim-1).expand(*spec.shape[:-1], self.steps, spec.shape[-1])
                observation_spec[out_key] = spec
            history_dict[out_key] = list(spec.zero().unbind(-2))
        self.history_dict = history_dict
        self.arange = torch.arange(self.steps, device=observation_spec.device).flip(0)
        return observation_spec
    
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            is_tuple = isinstance(in_key, tuple)
            if in_key in tensordict.keys(include_nested=is_tuple):
                item = tensordict.get(in_key)
                history = self.history_dict[out_key]
                history.append(item)
                history.pop(0)
        return tensordict
    
    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        step_count = tensordict.get("step_count")
        history_mask = (self.arange < step_count).float()
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if out_key in self.history_dict:
                observation = torch.stack(self.history_dict[out_key], dim=-2)
                mask_shape = (-1,) + (1,) * (observation.ndim-3) + (self.steps, 1)
                observation = observation * history_mask.reshape(mask_shape)
                tensordict.set(
                    out_key,
                    observation,
                )
        return tensordict


import torch.nn as nn
import torch.nn.functional as F

def make_mlp(units):
    layers = []
    for i, u in enumerate(units):
        layers.append(nn.LazyLinear(u))
        if i < len(units)-1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)

class AdaptationModule(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
    
    def forward(self, adapt_obs):

        return x

class Adaptation(Transform):

    def __init__(
        self, 
        obs_key: str,
        extrinsic_key: str,
        adapt_obs_key: str,
        encoder_units: Sequence[int],
        adapt_cfg,
        embed_dim: int=8,
    ):
        super().__init__([obs_key, extrinsic_key, adapt_obs_key], [obs_key])
        self.obs_key = obs_key
        self.extrinsic_key = extrinsic_key
        self.adapt_obs_key = adapt_obs_key
        self.initialized = False
        
        self.embed_dim = embed_dim
        self.encoder = make_mlp(list(encoder_units) + [embed_dim])
        self.adapt_module = AdaptationModule(adapt_cfg)
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        obs_spec = observation_spec[self.obs_key]
        if isinstance(obs_spec, CompositeSpec):
            raise TypeError
        spec = UnboundedContinuousTensorSpec(
            shape=[*obs_spec.shape[:-1], obs_spec.shape[-1]+self.embed_dim],
            device=obs_spec.device
        )
        observation_spec[self.obs_key] = spec
        return observation_spec
    
    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.training:
            extrinsics = tensordict.get(self.extrinsic_key)
            obs = tensordict.get(self.obs_key)
            embed = self.encoder(extrinsics)
            tensordict.set(self.obs_key, torch.cat([obs, embed], dim=-1))
        else:
            raise
        return tensordict
    
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs = tensordict.get(self.obs_key)
        tensordict.set(self.obs_key, obs[..., :-self.embed_dim])
        return tensordict
    
    forward = _call
        
    def train_adapt_module(self, tensordict):

        extrinsics = tensordict.get(self.extrinsic_key)
        adapt_obs = tensordict.get(self.adapt_obs_key)
        with torch.no_grad():
            embed = self.encoder(extrinsics)
        embed_pred = self.adapt_module(adapt_obs)
        loss = F.mse_loss(embed_pred, embed)