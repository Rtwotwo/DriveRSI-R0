"""Compatibility entry; implementation: driversi.data.datasets.history_windows."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.data.datasets.history_windows", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.data.datasets.history_windows")
