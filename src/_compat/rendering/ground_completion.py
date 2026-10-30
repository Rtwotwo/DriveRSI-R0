"""Compatibility entry; implementation: driversi.rendering.gaussian.ground_completion."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.rendering.gaussian.ground_completion", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.rendering.gaussian.ground_completion")
