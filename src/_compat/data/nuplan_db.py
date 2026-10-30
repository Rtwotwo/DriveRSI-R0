"""Compatibility entry; implementation: driversi.data.sources.nuplan_db."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.data.sources.nuplan_db", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.data.sources.nuplan_db")
