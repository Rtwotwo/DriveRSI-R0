"""Deterministic dependency compilation for bounded Harness tasks."""
from driversi.rsi.contracts import _identity, _keys


def compile_steps(rows, providers, *, allow_forward):
    nodes, position = {}, {}
    for index, row in enumerate(rows):
        _keys(row, ("id", "provider", "inputs", "config", "max_wall_seconds"), "Harness stage")
        _identity(row["id"], "stage ID")
        if row["id"] in nodes:
            raise ValueError("Duplicate stage identity")
        provider = providers.get(row["provider"])
        if not isinstance(row["inputs"], dict) or set(row["inputs"]) != dict(provider.inputs).keys():
            raise ValueError("Provider input ports do not match")
        nodes[row["id"]], position[row["id"]] = (row, provider), index
    dependencies = {}
    for identity, (row, provider) in nodes.items():
        deps = []
        for port, kind in provider.inputs:
            source = row["inputs"][port]
            if not isinstance(source, str) or source not in nodes:
                raise ValueError("Missing stage dependency: " + port)
            if nodes[source][1].output != kind:
                raise ValueError("Incompatible stage dependency: " + port)
            if not allow_forward and position[source] >= position[identity]:
                raise ValueError("Forward stage dependency requires schema version 2")
            deps.append(source)
        dependencies[identity] = sorted(set(deps), key=position.get)
    ordered, pending = [], list(nodes)
    while pending:
        ready = next((key for key in pending if all(dep in ordered for dep in dependencies[key])), None)
        if ready is None:
            raise ValueError("Cycle in Harness dependency graph")
        ordered.append(ready)
        pending.remove(ready)
    return [nodes[key][0] for key in ordered], dependencies


def blocked_by(row, stages):
    outcomes = {s["id"]: s["status"] for s in stages}
    return {source: outcomes.get(source, "not_executed") for source in row["inputs"].values()
            if outcomes.get(source) != "completed"}
