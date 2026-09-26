"""Reflex-Agent UI Package."""
try:
    import ctranslate2
except ImportError:
    pass

def launch_hud(*args, **kwargs):
    from .floating_hud import launch_hud as _launch
    return _launch(*args, **kwargs)

__all__ = ["launch_hud"]
