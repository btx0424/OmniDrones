from .control.hover import Hover
from .control.prey import Prey

def make(cfg, headless):
    # return Hover(cfg, headless)
    return Prey(cfg, headless)