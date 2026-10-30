"""Compatibility entry; implementation: driversi.agents.interfaces.catalog."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.agents.interfaces.catalog", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.agents.interfaces.catalog")
