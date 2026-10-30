"""Compatibility entry; implementation: driversi.rendering.inspection.case_gallery."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.rendering.inspection.case_gallery", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.rendering.inspection.case_gallery")
