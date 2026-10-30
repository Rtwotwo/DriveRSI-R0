"""Compatibility entry; implementation: driversi.sim.persistence.recording."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.sim.persistence.recording", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.sim.persistence.recording")
