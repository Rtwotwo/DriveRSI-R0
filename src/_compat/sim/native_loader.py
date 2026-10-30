"""Compatibility entry; implementation: driversi.sim.backends.native_loader."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.sim.backends.native_loader", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.sim.backends.native_loader")
