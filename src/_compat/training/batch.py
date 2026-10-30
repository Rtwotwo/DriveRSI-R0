"""Compatibility entry; implementation: driversi.training.sampling.batch."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.training.sampling.batch", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.training.sampling.batch")
