from .control.hover import Hover

def make(cfg, headless):
    return Hover(cfg, headless)