"""Compatibility entry; implementation: driversi.runtime.budget."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.runtime.budget", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.runtime.budget")
