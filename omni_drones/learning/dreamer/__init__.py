import torch
from tensordict import TensorDict

from . import models

class DreamerPolicy(object):
    def __init__(
        self,
        cfg,
        agent_spec,
    ):
        self.cfg = cfg
        self.agent_spec = agent_spec

        self._wm = models.WorldModel()
        self._task_behavior = models.ImagBehavior()

        if cfg.expl_behavior == "greedy":
            self._expl_behavior = self._task_behavior
        else:
            raise NotImplementedError(cfg.expl_behavior)
        
    def __call__(self, tensordict: TensorDict):
        embed = self._wm.encoder(obs)

    def train_op(self):
        ...