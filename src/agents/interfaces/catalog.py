"""Discover named agent configurations without importing model implementations."""
import re
from pathlib import Path

from driversi.agents.interfaces.config import read_agent_config


def agent_configs(directory=None):
    root = Path(directory) if directory is not None else Path(__file__).resolve().parents[3] / "src/config/agents"
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Agent config directory missing: {root}; supply --agent-dir")
    entries = {}
    for path in sorted(root.iterdir()):
        if path.name in ("legacy-status.json", "native-sources.lock.json"):
            continue
        if not path.is_file() or path.suffix not in (".json", ".yaml", ".yml"):
            continue
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", path.stem):
            raise ValueError(f"Agent config stem must be a simple name: {path.name}")
        if path.stem in entries:
            raise ValueError(f"Ambiguous agent name {path.stem!r}: {entries[path.stem]} and {path}")
        entries[path.stem] = path
    return entries


def resolve_agent(name, directory=None):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", name):
        raise ValueError("Agent name must be a simple name without path or extension")
    entries = agent_configs(directory)
    if name not in entries:
        raise ValueError(f"Unknown agent {name!r}; available: {', '.join(entries)}")
    return entries[name]


def list_agents(directory=None):
    return [{"name": name, "path": str(path), "config": read_agent_config(path)}
            for name, path in agent_configs(directory).items()]
