"""Compatibility entry; implementation: driversi.agents.visual_control.image_moment."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.agents.visual_control.image_moment", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.agents.visual_control.image_moment")
