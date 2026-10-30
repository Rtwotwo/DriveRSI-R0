"""Read-only service catalog over actual DriveHarness providers."""
from driversi.config import load
from driversi.rsi.workflow.providers import builtin_providers


def describe(group=None):
    definition = load("rsi/registry/services.json")
    groups = definition["groups"]
    if group is not None and group not in groups:
        raise ValueError("Unknown service group: " + str(group))
    registry = builtin_providers()
    registered = {row["name"]: row for row in registry.describe()}
    names = [name for members in groups.values() for name in members]
    if len(names) != len(set(names)) or set(names) != set(registered):
        raise ValueError("Service catalog must classify every registered provider exactly once")
    result = []
    for category, members in groups.items():
        if group is not None and group != category:
            continue
        for name in members:
            provider = registry.get(name)
            result.append({**registered[name], "group": category,
                           "prepare": provider.prepare.__module__ + "." + provider.prepare.__name__,
                           "execute": provider.execute.__module__ + "." + provider.execute.__name__,
                           "dispatcher": "driversi.rsi.workflow.engine",
                           "readiness": "registered_requires_task_preparation"})
    return result
