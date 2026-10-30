"""Compatibility entry; implementation: driversi.data.datasets.tto_audit."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.data.datasets.tto_audit", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.data.datasets.tto_audit")
