"""Create explicit full-population batches without starting GPU work."""

import json
from pathlib import Path

from .store import encode


def plan_batches(catalog, purpose, config, output, mode, batch_size=512):
    if mode not in ("embed", "label") or type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise ValueError("Use embed/label and batches of 1..1000")
    config = Path(config).resolve(strict=True)
    json.loads(config.read_text())
    sql, params = catalog.token_query(purpose)
    tokens = [r[0] for r in catalog.db.execute(sql, params)]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for i, start in enumerate(range(0, len(tokens), batch_size)):
        selected = tokens[start:start + batch_size]
        token_file = output / f"tokens-{i:04d}.json"
        token_file.write_text(encode(selected))
        argv = ["scripts/data/manage_scene_catalog.py", mode, "--database", str(catalog.path.resolve()),
                "--split", catalog.split, "--config", str(config), "--tokens", str(token_file),
                "--output", str(output / f"run-{i:04d}"), "--limit", str(len(selected)),
                "--max-seconds", "3600"]
        jobs.append({"batch": i, "tokens": len(selected), "argv": argv})
    plan = {"kind": "planned_not_executed", "split": catalog.split, "purpose": purpose,
            "mode": mode, "scene_count": len(tokens), "batch_size": batch_size, "jobs": jobs,
            "execution": "Run each argv with the allocated local model Python; inspect complete and failures; rerun incomplete batches.",
            "trainable": False}
    (output / "plan.json").write_text(encode(plan))
    return plan
