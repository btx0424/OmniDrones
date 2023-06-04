from typing import Optional, Sequence, Dict
from tensordict.tensordict import TensorDictBase

from torchrl.data.tensor_specs import TensorSpec
import torch
import torch.nn as nn

from tensordict import TensorDict
from torchrl.envs.transforms import Transform

class DynamicsModel(Transform):
    def __init__(self, cfg, obs_key, action_key):
        super().__init__()
        self.cfg = cfg
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:

        return observation_spec
    
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        
        return tensordict
    
    def forward(self, obs, action):
        ...

    def train_op(self, tensordict: TensorDict) -> Dict:
        ...


class LatentDynamicsModel(Transform):
    def __init__(self, cfg, obs_key, action_key):
        super().__init__()
        self.cfg = cfg
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:

        return observation_spec
    
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:

        return tensordict
    
    def forward(self, obs, action):
        ...

    def train_op(self, tensordict: TensorDict) -> Dict:
        ...


class LatentConsistencyModel(Transform):
    def __init__(self, cfg, obs_key, action_key):
        super().__init__()
        self.cfg = cfg
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:

        return observation_spec
    
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        
        return tensordict
    
    def forward(self, obs, action):
        ...

    def train_op(self, tensordict: TensorDict) -> Dict:
        ...