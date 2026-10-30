"""Read plain JSON/YAML mappings, without Hydra interpolation or object construction."""
import json
from pathlib import Path


PROJECT_ROOT_TOKEN = "@DRIVERSI_ROOT@"


def project_paths(value):
    """Resolve only the release's checkout token; never interpolate environment or code."""
    if isinstance(value, dict):
        return {key: project_paths(child) for key, child in value.items()}
    if isinstance(value, list):
        return [project_paths(child) for child in value]
    if isinstance(value, str) and PROJECT_ROOT_TOKEN in value:
        return value.replace(PROJECT_ROOT_TOKEN, str(Path(__file__).resolve().parent.parent))
    return value


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError(f"Duplicate or non-string config key: {key!r}")
        result[key] = value
    return result


def _plain(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Config keys must be strings")
        for child in value.values():
            _plain(child)
    elif isinstance(value, list):
        for child in value:
            _plain(child)
    elif isinstance(value, str) and "${" in value:
        raise ValueError("Hydra interpolation is not supported; supply an explicit value")


def read_mapping(path: Path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text, object_pairs_hook=_unique_pairs)
    elif path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configuration requires PyYAML>=6,<7") from exc

        class UniqueLoader(yaml.SafeLoader):
            pass

        def mapping(loader, node):
            return _unique_pairs((loader.construct_object(k), loader.construct_object(v))
                                 for k, v in node.value)

        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        try:
            value = yaml.load(text, Loader=UniqueLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML configuration: {path}") from exc
    else:
        raise ValueError("Config must have a .json, .yaml or .yml extension")
    if not isinstance(value, dict):
        raise ValueError("Config must be a mapping")
    try:
        # Restrict YAML to the same finite, serializable values accepted by JSON.
        value = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Config must contain finite JSON-compatible values") from exc
    _plain(value)
    return project_paths(value)
