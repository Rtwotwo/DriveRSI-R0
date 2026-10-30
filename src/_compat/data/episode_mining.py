"""Compatibility entry; implementation: driversi.data.scenes.episode_mining."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.data.scenes.episode_mining", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.data.scenes.episode_mining")
