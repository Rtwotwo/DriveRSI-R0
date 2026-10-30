"""Fresh-process dispatch to an owned model training adapter."""

import importlib
import sys
from pathlib import Path

# Old native environments retain their released dependencies. File invocation
# installs only owned namespace locations; it does not import another NAVSIM.
if __package__ in (None, ""):
    import types

    root = Path(__file__).resolve().parents[3]
    for name, folder in (("driversi", "src"), ("driversi_agents", "agents")):
        module = types.ModuleType(name)
        module.__path__ = [str(root / folder)]
        sys.modules[name] = module

from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json


def main():
    plan = read_mapping(sys.argv[1])
    result = importlib.import_module(plan["worker"]).run(plan)
    if result is not None:
        write_json(Path(plan["output"]) / "worker-result.json", result)


if __name__ == "__main__":
    main()
