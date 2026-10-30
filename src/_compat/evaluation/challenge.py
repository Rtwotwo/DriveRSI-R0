"""Compatibility entry; implementation: driversi.evaluation.search.challenge."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.evaluation.search.challenge", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.evaluation.search.challenge")
