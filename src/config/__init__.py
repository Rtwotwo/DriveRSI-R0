"""Versioned platform presets; experiment outputs and credentials are not defaults."""
import hashlib
from pathlib import Path


def root():
    return Path(__file__).resolve().parent


def path(name):
    name = str(name)
    relative = Path(name)
    if not name or any(p.startswith(".") or p == "local" for p in relative.parts) or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("A relative preset ID without traversal is required")
    result = (root() / relative).resolve()
    if not result.is_relative_to(root()) or not result.is_file():
        raise ValueError("Unknown platform preset: " + name)
    return result


def load(name):
    """Read a preset resource; executable tasks apply their own strict schemas."""
    preset = path(name)
    content = preset.read_text(encoding="utf-8")
    if preset.suffix == ".json":
        import json
        from driversi.configuration import project_paths
        return project_paths(json.loads(content))
    if preset.suffix in {".yaml", ".yml"}:
        import yaml
        from driversi.configuration import project_paths
        return project_paths(yaml.safe_load(content))
    if preset.suffix == ".toml":
        import tomllib
        return tomllib.loads(content)
    if preset.suffix in {".txt", ".md"}:
        return content
    raise ValueError("Unsupported preset resource format: " + preset.suffix)


def catalog(group=None):
    rows = []
    for item in sorted(root().rglob("*")):
        relative = item.relative_to(root())
        if not item.is_file() or item.suffix not in {".json", ".yaml", ".yml", ".toml", ".txt", ".md"}:
            continue
        if any(p.startswith(".") or p == "local" for p in relative.parts):
            continue
        category = relative.parts[0] if len(relative.parts) > 1 else "platform"
        if group is not None and category != group:
            continue
        rows.append({"id": relative.as_posix(), "group": category, "path": str(item),
                     "sha256": hashlib.sha256(item.read_bytes()).hexdigest()})
    return rows
