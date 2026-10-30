"""Exact released benchmark token coverage; no replacement or silent subset."""

import concurrent.futures
import json
import sqlite3
from pathlib import Path

import yaml


def scan(item):
    path, tokens = item
    placeholders = ",".join("?" for _ in tokens)
    with sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT token FROM lidar_pc WHERE token IN (" + placeholders + ")",
            [bytes.fromhex(t) for t in tokens],
        ).fetchall()
    return str(path), [bytes(v[0]).hex() for v in rows]


def main():
    root = Path(__file__).resolve().parents[2]
    repo = root / "agents/driverl/native/DriveRL"
    out = root / "outputs/quality/driverl-benchmark-v1"
    out.mkdir(parents=True, exist_ok=True)
    data = Path("/data/xinyao/dataset/nuPlan/nuplan-v1.1/splits")
    results = {}
    for name, folder, expected in [
        ("driverl_val14", "val", 1118),
        ("driverl_test14_hard", "test", 272),
        ("driverl_test14_random", "test", 261),
    ]:
        config = yaml.safe_load(
            (
                repo
                / "nuplan-devkit/nuplan/planning/script/config/common/scenario_filter"
                / (name + ".yaml")
            ).read_text()
        )
        tokens = config["scenario_tokens"]
        assert len(tokens) == len(set(tokens)) == expected
        matches = {}
        errors = []
        paths = sorted((data / folder).glob("*.db"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(scan, (p, tokens)) for p in paths]
            for n, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    path, found = future.result()
                    for token in found:
                        matches.setdefault(token, []).append(path)
                except Exception as e:
                    errors.append(repr(e))
                if n % 100 == 0:
                    print(name, n, len(paths), len(matches), flush=True)
        missing = sorted(set(tokens) - set(matches))
        duplicates = {t: v for t, v in matches.items() if len(v) != 1}
        entry = dict(
            expected=expected,
            found=len(matches),
            missing=missing,
            duplicates=duplicates,
            errors=errors,
            matches=matches,
            complete=not missing and not duplicates and not errors,
        )
        if entry["complete"]:
            links = out / "db-links" / name
            links.mkdir(parents=True, exist_ok=True)
            for file in sorted({v[0] for v in matches.values()}):
                target = links / Path(file).name
                if not target.exists():
                    target.symlink_to(file)
        results[name] = entry
        (out / "coverage.json").write_text(json.dumps(results, indent=2))
        print(name, entry["complete"], len(matches), flush=True)


if __name__ == "__main__":
    main()
