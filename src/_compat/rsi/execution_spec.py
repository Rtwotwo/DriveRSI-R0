"""Compatibility entry; implementation: driversi.rsi.execution.execution_spec."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.rsi.execution.execution_spec", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.rsi.execution.execution_spec")
