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


from .mappo import MAPPOPolicy
from .mappo_new import MAPPO
from .ppo import *
from .happo import HAPPOPolicy
from .qmix import QMIXPolicy

from .dqn import DQNPolicy
from .sac import SACPolicy
from .td3 import TD3Policy
from .matd3 import MATD3Policy
from .tdmpc import TDMPCPolicy

ALGOS = {
    "mappo_old": MAPPOPolicy,
    "mappo": MAPPO,
    "happo": HAPPOPolicy,
    "ppo": PPOPolicy,
    "ppo_rnn": PPORNNPolicy,
    "ppo_adapt": PPOAdaptivePolicy,
    "sac": SACPolicy,
    "td3": TD3Policy,
}
