"""Compatibility entry; implementation: driversi.evaluation.benchmarks.reactive."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.evaluation.benchmarks.reactive", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.evaluation.benchmarks.reactive")
