"""Compatibility entry; implementation: driversi.rendering.gaussian.surface_quality."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.rendering.gaussian.surface_quality", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.rendering.gaussian.surface_quality")
