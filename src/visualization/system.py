"""Curated architecture projection with links into registered aggregate evidence."""

from .catalog import read_json, safe_text

STATUSES = {"pilot", "partial", "planned", "active"}
SECTIONS = {
    "steps": ("id", "title", "short", "subtitle", "status", "inputs", "operation", "outputs", "acceptance", "current", "evidence"),
    "layers": ("id", "title", "owner", "status", "description", "outputs", "evidence"),
    "models": ("id", "title", "role", "status", "detail"),
    "priorities": ("id", "title", "status", "detail", "evidence"),
}


def system_overview(config, project):
    relative = config.get("architecture")
    if not relative:
        return None
    root = (project / "src/config/visualization").resolve()
    path = (project / relative).resolve()
    if not path.is_relative_to(root) or path.suffix != ".json":
        raise ValueError("Architecture must be an owned visualization preset")
    source = read_json(path)
    if source.get("schema_version") != 1:
        raise ValueError("Unsupported architecture schema")
    evidence = {
        f"{kind}:{row['id']}"
        for kind, collection in (("experiment", "experiments"), ("production", "production"))
        for row in config.get(collection, [])
    }
    result = {key: safe_text(source[key]) for key in ("reviewed_on", "summary", "boundary")}
    for section, fields in SECTIONS.items():
        rows = source.get(section, [])
        if not isinstance(rows, list) or len(rows) > 30:
            raise ValueError("Invalid architecture section")
        result[section] = []
        seen = set()
        for row in rows:
            if set(row) != set(fields) or not all(isinstance(row[key], str) for key in fields):
                raise ValueError("Architecture accepts only declared text fields")
            if row["status"] not in STATUSES or row["id"] in seen:
                raise ValueError("Invalid architecture state or duplicate id")
            if row.get("evidence") and row["evidence"] not in evidence:
                raise ValueError("Architecture references unregistered evidence")
            seen.add(row["id"])
            result[section].append({key: safe_text(row[key]) for key in fields})
    contracts = source.get("contracts", [])
    if not isinstance(contracts, list) or len(contracts) > 20 or not all(isinstance(x, str) for x in contracts):
        raise ValueError("Invalid publication contracts")
    result["contracts"] = [safe_text(x) for x in contracts]
    return result
