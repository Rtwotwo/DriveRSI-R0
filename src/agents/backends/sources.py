"""Owned model sources: upstream provenance without a runtime Git/clone dependency."""

import hashlib
import json
import sys
from pathlib import Path


def source_root(path):
    path = Path(path).resolve()
    for parent in (path, *path.parents):
        if (parent / "source-manifest.json").is_file():
            return parent
    raise ValueError("Owned source manifest missing: " + str(path))


def source_state(path, expected=None):
    root = source_root(path)
    manifest = root / "source-manifest.json"
    value = json.loads(manifest.read_text())
    if value.get("schema_version") != 1 or not isinstance(value.get("files"), dict):
        raise ValueError("Invalid owned source manifest")
    files = {}
    for name in value["files"]:
        p = root / name
        if not p.resolve().is_relative_to(root) or p.is_symlink():
            raise ValueError("Owned source escapes its directory")
        files[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    extras = [str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json", ".sh")
              and p != manifest and str(p.relative_to(root)) not in files
              and not any(part in ("__pycache__", ".pytest_cache", ".ruff_cache")
                          for part in p.relative_to(root).parts)]
    return {"expected": expected or value["revision"], "actual": value["revision"],
            "clean": files == value["files"] and not extras,
            "root": str(root), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "content_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "extra_sources": extras}


def require_source(path, expected=None):
    state = source_state(path, expected)
    if state["actual"] != state["expected"] or not state["clean"]:
        raise ValueError("Owned model source changed or revision mismatch: " + str(path))
    return state


def require_model_root(path, model):
    project = Path(__file__).resolve().parents[3]
    expected = project / "agents" / model / "native"
    if Path(path).resolve() != expected:
        raise ValueError("Model implementation must be owned under agents/<model>/native")
    return expected


def activate_packages(locations):
    """Select owned package roots without overwriting a previously loaded SDK variant."""
    for name, location in locations.items():
        module = sys.modules.get(name)
        if module is not None:
            origin = getattr(module, "__file__", None)
            if not origin or not Path(origin).resolve().is_relative_to(location):
                raise ValueError("Conflicting model runtime already imported: " + name)
    for location in reversed(list(locations.values())):
        path = str(location.parent)
        if path not in sys.path:
            sys.path.insert(0, path)


def activate_driverl():
    """Keep original package names/checkpoint keys in the isolated model process."""
    root = Path(__file__).resolve().parents[3]
    native = root / "agents/driverl/native/DriveRL"
    activate_packages({"driverl": native / "src/driverl",
                       "nuplan": native / "nuplan-devkit/nuplan"})
    return native
