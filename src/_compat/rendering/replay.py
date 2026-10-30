"""Compatibility entry; implementation: driversi.rendering.bev.replay."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.rendering.bev.replay", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.rendering.bev.replay")
