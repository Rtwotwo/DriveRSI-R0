"""Versioned NAVSIM training bridge; upstream model, losses and optimizer stay native."""

import json
from pathlib import Path

import yaml


def validate_dataset(manifest_path, data_root, official_test_filter):
    """Admit original NAVSIM sources; synthetic teacher data needs a separate gate."""
    path = Path(manifest_path)
    value = json.loads(path.read_text())
    keys = {"schema_version", "dataset_version", "dataset_kind", "data_root", "train_filter"}
    if set(value) != keys or value["schema_version"] != 1 or not value["dataset_version"]:
        raise ValueError("Malformed native training dataset manifest")
    if value["dataset_kind"] != "original_navsim":
        raise ValueError("Synthetic teacher admission is not connected to native training")
    if Path(value["data_root"]).resolve() != Path(data_root).resolve():
        raise ValueError("Training manifest data root differs from configured data")
    train_filter = Path(value["train_filter"])
    if not train_filter.is_absolute():
        train_filter = path.parent / train_filter
    train_filter = train_filter.resolve(strict=True)
    train = yaml.safe_load(train_filter.read_text())
    test = yaml.safe_load(Path(official_test_filter).read_text())
    names = train.get("log_names")
    withheld = test.get("log_names")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
        raise ValueError("Training filter must enumerate source logs")
    if len(names) != len(set(names)) or not isinstance(withheld, list) or not withheld:
        raise ValueError("Invalid source isolation filters")
    from driversi.training.sampling.environment import source_log_group

    if {source_log_group(n) for n in names} & {source_log_group(n) for n in withheld}:
        raise ValueError("Training logs overlap official navtest logs")
    for name in names:
        if (
            Path(name).name != name
            or not (Path(data_root) / "navsim_logs/trainval" / (name + ".pkl")).is_file()
        ):
            raise ValueError("Training source log missing or invalid: " + name)
    return train_filter
