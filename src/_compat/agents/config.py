"""Compatibility entry; implementation: driversi.agents.interfaces.config."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.agents.interfaces.config", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.agents.interfaces.config")
