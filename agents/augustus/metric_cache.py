"""Resolve exactly the metadata paths consumed by the bundled NAVSIM loader."""

from pathlib import Path


def scoring_files(cache, records):
    root = Path(cache).resolve()
    metadata = root / "metadata"
    if not metadata.is_dir():
        return [str(metadata / "required.csv")]
    csvs = [p for p in metadata.iterdir() if ".csv" in str(p)]
    if not csvs:
        return [str(metadata / "required.csv")]
    if len(csvs) != 1 or not csvs[0].is_file():
        raise ValueError("Native scoring requires one unambiguous metadata CSV")
    mapping = {}
    for line in csvs[0].read_text().splitlines()[1:]:
        path = Path(line)
        if not path.is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError("Native metadata points outside configured score context")
        if path.name != "metric_cache.pkl" or len(path.parts) < 4:
            raise ValueError("Malformed native metric path")
        token = path.parent.name
        if token in mapping:
            raise ValueError("Duplicate scoring token")
        mapping[token] = path
    files = [str(csvs[0])]
    for row in records:
        path = mapping.get(row["frame_id"])
        if path is None:
            raise ValueError("Required evaluation token missing from metadata")
        if path.parents[2].name != row["log_name"]:
            raise ValueError("Metric cache source log differs from declared anchor")
        files.append(str(path))
    return files
