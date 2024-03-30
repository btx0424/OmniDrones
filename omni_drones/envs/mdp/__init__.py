import inspect
from .mdp_term import TerminationFunc, RewardFunc
from .action import *
from .observation import *
from .randomization import *

def _is_subclass(cls, parent):
    return inspect.isclass(cls) and issubclass(cls, parent)

OBS_FUNCS = {k: v for k, v in vars(observation).items() if _is_subclass(v, ObservationFunc)}
TERM_FUNCS = {k: v for k, v in vars(mdp_term).items() if _is_subclass(v, TerminationFunc)}
REW_FUNCS = {k: v for k, v in vars(mdp_term).items() if _is_subclass(v, RewardFunc)}
ACT_FUNCS = {k: v for k, v in vars(action).items() if _is_subclass(v, ActionFunc)}
RAND_FUNCS = {k: v for k, v in vars(randomization).items() if _is_subclass(v, Randomization)}