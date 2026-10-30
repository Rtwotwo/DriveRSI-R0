"""DriveTTO proposals are data, not a replacement for Harness or the evaluator."""

from pathlib import Path

import numpy as np

from driversi.data.datasets.tto_audit import COMPONENTS, json_read, scene_path, sha256, verified_rows
from driversi.runtime.budget import write_json
from driversi.rsi.contracts import _keys
from driversi.training.sampling.environment import source_log_group


def prepare_candidates(config, base, registry):
    _keys(config, ("bank", "cases", "protected_logs", "max_scenes"), "DriveTTO import")
    if type(config["max_scenes"]) is not int or not 1 <= config["max_scenes"] <= 10000:
        raise ValueError("Bounded scene count required")
    paths = {
        k: str((base / config[k]).resolve(strict=True)) for k in ("bank", "cases", "protected_logs")
    }
    cases = json_read(paths["cases"])
    protected = json_read(paths["protected_logs"])
    if not isinstance(protected, list) or any(not isinstance(x, str) or not x for x in protected):
        raise ValueError("Explicit protected source logs required")
    if not isinstance(cases, list) or not 1 <= len(cases) <= config["max_scenes"]:
        raise ValueError("Cases exceed declared import budget")
    tokens = set()
    for case in cases:
        _keys(case, ("split", "token"), "DriveTTO case")
        scene_path(paths["bank"], case["split"], case["token"])
        if case["token"] in tokens:
            raise ValueError("Duplicate case")
        tokens.add(case["token"])
    return {
        **config,
        **paths,
        "cases_sha256": sha256(paths["cases"]),
        "protected_sha256": sha256(paths["protected_logs"]),
    }, 0


def collect(config, protected_extra=(), *, allowed_navval=None):
    for name in ("cases", "protected"):
        path = config["cases" if name == "cases" else "protected_logs"]
        if sha256(path) != config[name + "_sha256"]:
            raise ValueError("Declared DriveTTO input changed")
    protected = set(json_read(config["protected_logs"])) | set(protected_extra)
    protected = {source_log_group(x) for x in protected}
    accepted, rejected, sources = [], [], set()
    for case in json_read(config["cases"]):
        folder = scene_path(config["bank"], case["split"], case["token"])
        rows, digest = verified_rows(folder)
        contract = json_read(folder / "contract.json")
        identity, protocol = contract["identity"], contract["protocol"]
        if any(identity[k] != case[k] for k in ("split", "token")):
            raise ValueError("DriveTTO scene identity mismatch")
        if tuple(identity["component_names"]) != COMPONENTS:
            raise ValueError("Unknown DriveTTO score components")
        construction, labeling = protocol["constructor"], protocol["labeling"]
        if (
            construction["num_poses"],
            construction["interval_s"],
            construction["coordinate_frame"],
        ) != (8, 0.5, "current_ego_x_forward_y_left"):
            raise ValueError("Unsupported candidate trajectory contract")
        if (labeling["protocol"], labeling["evaluation_group"], labeling["score_unit"]) != (
            "navsim_v1_official_single_candidate",
            "reference_plus_one_candidate",
            "raw_0_to_1",
        ):
            raise ValueError("Unsupported independent score context")
        if not identity["log_name"] or Path(identity["log_name"]).name != identity["log_name"]:
            raise ValueError("Invalid source log")
        best = int(np.argmax(rows["labels"][:, -1]))
        reason = None
        if source_log_group(identity["log_name"]) in protected:
            reason = "protected_source"
        elif case["split"] != "train" and not (
            case["split"] == "val" and allowed_navval is not None
            and allowed_navval.get(case["token"]) == identity["log_name"]
        ):
            reason = "validation_source_not_training"
        else:
            gt = np.asarray(identity["expert_gt"], dtype=np.float32)
            valid = [
                int(i)
                for i in np.argsort(-rows["labels"][:, -1], kind="stable")
                if rows["labels"][i, -1] >= 0.95 and not np.array_equal(rows["trajectories"][i], gt)
            ]
            if valid:
                best = valid[0]
            else:
                reason = "no_distinct_candidate_at_fixed_0.95_threshold"
        row = rows[best]
        evidence = {
            **case,
            "log_name": identity["log_name"],
            "timestamp_us": identity["timestamp_us"],
            "ego_pose": identity["ego_pose"],
            "expert_gt": identity["expert_gt"],
            "candidate_row": best,
            "trajectory": row["trajectories"].tolist(),
            "labels": dict(zip(COMPONENTS, row["labels"].tolist())),
            "table_rows": len(rows),
            "table": str(folder / "candidates.npy"),
            "table_sha256": digest,
            "contract": str(folder / "contract.json"),
            "contract_sha256": sha256(folder / "contract.json"),
            "publication": str(folder / "ALL_SCORES.json"),
            "publication_sha256": sha256(folder / "ALL_SCORES.json"),
            "reason": reason,
        }
        if reason:
            rejected.append(evidence)
        else:
            accepted.append(evidence)
            sources.add(identity["log_name"])
    return {
        "schema_version": 1,
        "kind": "original_anchor_proposals",
        "generator": "DriveTTO",
        "role": "trajectory_proposals_only",
        "accepted": accepted,
        "rejected": rejected,
        "source_groups": sorted(sources),
        "new_score_queries": 0,
        "new_rollouts": 0,
        "score_scope": "saved_official_v1_original_anchor_not_reactive",
        "threshold": 0.95,
        "reactive_trajectory_admitted": False,
        "visual_admitted": False,
        "trainable": False,
    }


def run_candidates(config, inputs, output, registry):
    from driversi.rsi.execution.registry import ExperimentRegistry

    report = collect(config, ExperimentRegistry(registry).protected_groups())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    return {
        "status": "completed",
        "reason": None,
        "sim_agent_steps": 0,
        "artifacts": {"report": str(output / "report.json")},
        "summary": report,
        "admission": {"trainable": False, "scope": report["score_scope"]},
    }
