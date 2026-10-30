"""Compatibility entry; implementation: driversi.rendering.execution.teacher_pilot."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.rendering.execution.teacher_pilot", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.rendering.execution.teacher_pilot")
