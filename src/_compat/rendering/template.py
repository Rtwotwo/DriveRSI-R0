"""Compatibility entry; implementation: driversi.rendering.bev.template."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.rendering.bev.template", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.rendering.bev.template")
