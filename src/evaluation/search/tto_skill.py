"""Isolated DriveTTO skill worker. Original constructor and scorer, bounded queries.

This file also runs under the original DriveTTO Python 3.9 environment. The
Harness owns admission and source isolation; this worker never trains a model.
"""

import argparse
import hashlib
import json
import lzma
import pickle
import sys
from pathlib import Path

import numpy as np

COMPONENTS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)
ROW_DTYPE = np.dtype(
    [
        ("keys", "<i4", (2,)),
        ("trajectories", "<f4", (8, 3)),
        ("labels", "<f8", (7,)),
        ("phase", "<i2"),
    ],
    align=True,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def scoring_coordinates(trajectory, yaw_delta):
    aligned = np.asarray(trajectory, dtype=np.float64).copy()
    c, s = np.cos(yaw_delta), np.sin(yaw_delta)
    aligned[:, :2] = aligned[:, :2] @ np.array([[c, s], [-s, c]])
    aligned[:, 2] = np.arctan2(np.sin(aligned[:, 2] + yaw_delta), np.cos(aligned[:, 2] + yaw_delta))
    return aligned


def execute(config, output, mode):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for path, sha in config["files"].items():
        if digest(path) != sha:
            raise ValueError("Skill input changed: " + path)
    sys.path.insert(0, str(Path(config["backend"]) / "feedback/src"))
    from dataclasses import asdict

    from drivetto_feedback.batch.source import Evaluator, construct

    spent = {"attempted_queries": 0, "completed_queries": 0, "agent_steps_per_query": 80}
    write(output / "cost.json", spent)

    source_yaw_delta = 0.0

    def score(evaluator, trajectory):
        if spent["attempted_queries"] >= config["max_queries"]:
            raise ValueError("Fixed official scoring budget exhausted")
        spent["attempted_queries"] += 1
        write(output / "cost.json", spent)  # Failed/unknown queries remain charged.
        # Source cache uses quaternion heading; native nuPlan rear axle may use
        # projected yaw. Transform the entire trajectory into the evaluator's
        # declared frame instead of silently relaxing anchor equality.
        labels = evaluator.score(scoring_coordinates(trajectory, source_yaw_delta))
        spent["completed_queries"] += 1
        write(output / "cost.json", spent)
        return labels

    reports = []
    for case in config["cases"]:
        contract = read(case["contract"])
        identity, protocol = contract["identity"], contract["protocol"]
        with lzma.open(identity["official_cache_path"], "rb") as stream:
            cache = pickle.load(stream)  # Explicit, sealed, trusted local native cache.
        pose = cache.ego_state.rear_axle
        if (
            cache.ego_state.time_point.time_us != identity["timestamp_us"]
            or not np.allclose([pose.x, pose.y], identity["ego_pose"][:2], atol=1e-7, rtol=0)
            or not np.array_equal([pose.x, pose.y, pose.heading], identity["scoring_ego_pose"])
        ):
            raise ValueError("Original-anchor scoring cache mismatch")
        source_yaw_delta = identity["ego_pose"][2] - pose.heading
        if abs(source_yaw_delta) > 1e-3:
            raise ValueError(
                "Source/scoring orientation discrepancy is not a supported heading representation conversion"
            )
        evaluator = Evaluator(cache)
        if asdict(evaluator.scorer._config) != identity["pdm_config"]:
            raise ValueError("Native scoring protocol changed")
        gt = np.asarray(identity["expert_gt"], dtype=np.float64)
        if mode == "verify":
            row = case["selected"]
            table = np.load(row["table"], allow_pickle=False)
            stored = table[row["candidate_row"]]
            if not np.array_equal(
                stored["trajectories"], np.asarray(row["trajectory"], dtype=np.float32)
            ) or not np.array_equal(stored["labels"], [row["labels"][k] for k in COMPONENTS]):
                raise ValueError("Selected branch differs from sealed scored row")
            labels = score(evaluator, stored["trajectories"])
            parity = float(np.max(np.abs(labels - stored["labels"])))
            reports.append(
                {
                    "token": identity["token"],
                    "labels": labels.tolist(),
                    "parity_error": parity,
                    "accepted": bool(parity <= 1e-9 and labels[-1] >= 0.95),
                }
            )
            continue
        zero = construct(gt, [[0, 0]], protocol)[0][0]
        restored = zero.copy()
        restored[:, 2] = gt[:, 2]
        controls = {
            name: dict(zip(COMPONENTS, score(evaluator, trajectory).tolist()))
            for name, trajectory in (
                ("expert_gt", gt),
                ("zero_offset_reconstruction", zero),
                ("zero_offset_gt_heading_diagnostic_only", restored),
            )
        }
        folder = output / "scenes/train" / identity["token"]
        folder.mkdir(parents=True)
        rows, seen = [], set()
        for phase in (0, 1):
            center = np.asarray(
                [0, 0] if phase == 0 else rows[int(np.argmax([x[2][-1] for x in rows]))][0]
            )
            offsets = config["initial_keys"] if phase == 0 else config["refinement_keys"]
            keys = []
            for key in center + np.asarray(offsets, dtype=np.int32):
                key = tuple(map(int, key))
                if key != (0, 0) and key not in seen:
                    seen.add(key)
                    keys.append(key)
            if not keys:
                continue
            paths = construct(gt, keys, protocol)[0]
            for key, path in zip(keys, paths):
                labels = score(evaluator, path)
                rows.append((key, path, labels, phase))
                np.save(folder / "partial.npy", np.array(rows, dtype=ROW_DTYPE))
        table = np.array(rows, dtype=ROW_DTYPE)
        np.save(folder / "candidates.npy", table)
        write(folder / "contract.json", contract)
        write(folder / "state.json", {"count": len(table)})
        write(
            folder / "ALL_SCORES.json",
            {
                "format": "all_scored_candidates_v1",
                "passed": True,
                "token": identity["token"],
                "split": "train",
                "candidates": len(table),
                "data_digest": digest(folder / "candidates.npy"),
            },
        )
        reports.append(
            {
                "token": identity["token"],
                "controls": controls,
                "candidates": len(table),
                "best_score": float(table["labels"][:, -1].max()),
                "diagnosis": "controlled_comparison_not_causal_attribution",
                "privileged_gt_used": True,
                "online_policy": False,
                "source_to_scoring_yaw_radians": source_yaw_delta,
            }
        )
    write(
        output / "report.json",
        {
            "kind": "drivetto_skill_" + mode,
            "cases": reports,
            "cost": spent,
            "score_scope": "original_anchor_logged_world_only",
            "reactive_admission": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("mode", choices=("search", "verify"))
    args = parser.parse_args()
    execute(read(args.config), args.output, args.mode)
