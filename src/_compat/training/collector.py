"""Compatibility entry; implementation: driversi.training.sampling.collector."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.training.sampling.collector", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.training.sampling.collector")
