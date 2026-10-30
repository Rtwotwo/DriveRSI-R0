"""Compatibility entry; implementation: driversi.rendering.visual.visual_replay."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.rendering.visual.visual_replay", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.rendering.visual.visual_replay")
