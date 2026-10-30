"""Owned NAVSIM v2 scoring runtime. Import in a separate process from v1."""
from driversi.agents.backends.native import project_root
from driversi.agents.backends.sources import activate_packages

_root = project_root()
activate_packages({
    "navsim": _root / "src/vendor/driving/navsim-v2.2/navsim",
    "nuplan": _root / "src/vendor/driving/nuplan-devkit/nuplan",
})
