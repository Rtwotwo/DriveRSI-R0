"""Compatibility entry; implementation: driversi.agents.backends.native."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.agents.backends.native", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.agents.backends.native")
