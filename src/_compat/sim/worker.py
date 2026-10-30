"""Compatibility entry; implementation: driversi.sim.workers.worker."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.sim.workers.worker", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.sim.workers.worker")
