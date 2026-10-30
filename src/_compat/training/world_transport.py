"""Compatibility entry; implementation: driversi.training.sampling.world_transport."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.training.sampling.world_transport", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.training.sampling.world_transport")
