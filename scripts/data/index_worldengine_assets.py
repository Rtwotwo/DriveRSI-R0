"""Join the complete released WorldEngine index to local NAVSIM training logs."""

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from driversi.data.assets.worldengine import bundle_files, read_config_index


def run(archive, split, logs, bundles, output):
    train_logs = yaml.safe_load(split.read_text())["log_names"]
    report = read_config_index(archive, train_logs)
    local = {p.stem for p in logs.glob("*.pkl")}
    for row in report["rows"]:
        row["local_log_present"] = row["source_log"] in local
        if (bundles / row["asset_id"]).is_dir():
            row["files"] = bundle_files(bundles, row["asset_id"])
        row["local_bundle_complete"] = bool(row.get("files")) and all(
            x["present"] for x in row["files"].values()
        )
        row["probe_eligible"] = (
            row["source_eligible"] and row["local_log_present"] and row["local_bundle_complete"]
        )
    report["summary"] = {
        "released_configs": len(report["rows"]),
        "eligible_local_logs": len(
            {
                x["source_log"]
                for x in report["rows"]
                if x["source_eligible"] and x["local_log_present"]
            }
        ),
        "eligible_configs": sum(
            x["source_eligible"] and x["local_log_present"] for x in report["rows"]
        ),
        "probe_bundles": sum(x["probe_eligible"] for x in report["rows"]),
        "cities": dict(
            Counter(
                x["city"] for x in report["rows"] if x["source_eligible"] and x["local_log_present"]
            )
        ),
    }
    report["student"] = "chainflow_official_stage1"
    report["weakness_evidence"] = "must_be_measured_separately_asset_presence_is_not_a_failure"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("archive", "split", "logs", "bundles", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.archive, a.split, a.logs, a.bundles, a.output)
