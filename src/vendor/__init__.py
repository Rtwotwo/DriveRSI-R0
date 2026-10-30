"""Locate pinned upstream code without importing conflicting native packages."""
import json
from pathlib import Path


def catalog():
    return json.loads(Path(__file__).with_name("registry.json").read_text())["dependencies"]


def path(name, *, project=None):
    entries = catalog()
    if name not in entries:
        raise ValueError("Unknown platform dependency: " + str(name))
    base = (Path(project).resolve() / "src/vendor") if project is not None else Path(__file__).resolve().parent
    result = (base / entries[name]["directory"]).resolve()
    if not result.is_relative_to(base) or not (result / "source-manifest.json").is_file():
        raise ValueError("Pinned dependency source missing: " + str(name))
    return result


def verify(name, *, project=None):
    from driversi.agents.backends.sources import require_source
    return require_source(path(name, project=project), catalog()[name]["revision"])
