"""Compatibility entry; implementation: driversi.data.participants.traffic_continuity."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.data.participants.traffic_continuity", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.data.participants.traffic_continuity")
