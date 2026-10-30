"""Inspect functional ownership and legacy entry points without loading model dependencies."""

import argparse
import ast
import json
from pathlib import Path


def inventory():
    root = Path(__file__).parent
    layout = json.loads(root.joinpath("layout.json").read_text())
    missing = []
    for old in layout["migrated"]:
        target = layout["module_moves"][old]
        path = root.joinpath(*target.split(".")[1:]).with_suffix(".py")
        if not path.is_file():
            missing.append(str(path))
    return {
        **layout,
        "source_root": str(root),
        "missing_implementations": missing,
        "implemented_modules": len(layout["migrated"]),
        "compatibility": "legacy imports load the same canonical module lazily",
    }



def script_violations(repository=None):
    """Check explicit command ownership in a source checkout, without importing jobs."""
    repository = Path(repository) if repository else Path(__file__).resolve().parent.parent
    scripts = repository / "scripts"
    if not scripts.is_dir():
        return []  # Installed wheels do not contain source-checkout commands.
    errors = []
    try:
        entries = json.loads((scripts / "catalog.json").read_text())["entries"]
    except (OSError, ValueError, KeyError):
        return ["scripts: missing or invalid command catalog"]
    paths = [entry["path"] for entry in entries]
    actual = {p.relative_to(repository).as_posix() for p in scripts.rglob("*.py")
              if "__pycache__" not in p.parts}
    if len(paths) != len(set(paths)):
        errors.append("scripts: duplicate catalog paths")
    for path in sorted(actual - set(paths)):
        errors.append(path + ": missing command purpose")
    for path in sorted(set(paths) - actual):
        errors.append(path + ": catalog entry has no command")
    for entry in entries:
        path = Path(entry["path"])
        if len(path.parts) < 3 or path.parts[0] != "scripts":
            errors.append(str(path) + ": flat or unowned command")
        if not entry.get("purpose") or not entry.get("implementation"):
            errors.append(str(path) + ": missing purpose or implementation")
        expected = "historical" if "legacy" in path.parts else "requires_preflight"
        if entry.get("status") != expected:
            errors.append(str(path) + ": incorrect lifecycle status")
    return errors


def structure_violations():
    """Keep business logic in owned packages and old imports in compatibility code."""
    root = Path(__file__).parent
    layout = json.loads(root.joinpath("layout.json").read_text())
    moved = layout["module_moves"]
    errors = []
    for domain in ("agents", "data", "evaluation", "rendering", "rsi", "sim", "training"):
        allowed = {"__init__.py", "__main__.py"}
        if domain == "rsi":
            allowed.add("contracts.py")
        for path in root.joinpath(domain).glob("*.py"):
            if path.name not in allowed:
                errors.append(str(path.relative_to(root)) + ": unowned flat implementation")
    for path in root.rglob("*.py"):
        if "_compat" in path.relative_to(root).parts or path.relative_to(root).parts[0] == "vendor":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module] + [node.module + "." + a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for name in names:
                if name in moved:
                    errors.append(
                        str(path.relative_to(root))
                        + ":"
                        + str(node.lineno)
                        + ": legacy import "
                        + name
                    )
    return errors + script_violations()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = inventory()
    report["structure_violations"] = structure_violations() if args.check else []
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))
    if args.check and (report["missing_implementations"] or report["structure_violations"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
