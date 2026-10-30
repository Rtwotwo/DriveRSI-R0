"""Compatibility entry; implementation: driversi.data.road_network.lane_graph."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("driversi.data.road_network.lane_graph", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("driversi.data.road_network.lane_graph")
