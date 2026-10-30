"""Compatibility entry; implementation: driversi.training.self_play.opponents."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.training.self_play.opponents", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.training.self_play.opponents")
