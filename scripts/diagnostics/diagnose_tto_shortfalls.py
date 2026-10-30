"""Bounded official-score contrasts; never modifies a candidate bank or samples new goals.
Run in the original DriveScorer scoring environment, with its feedback/src on PYTHONPATH.
"""
import argparse
import hashlib
import json
import lzma
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.max_scenes <= 24 or args.output.exists():
        parser.error("Require a new output and 1..24 scenes")
    import drivetto_feedback.batch.source as source
    import navsim.agents.drive_scorer.endpoint.path_endpoint_surface as constructor
    from drivetto_feedback.batch.source import Evaluator, PDMScorerConfig, construct
    audit = json.loads(args.audit.read_text())
    records = [r for r in audit["scenes"] if r["category"] in ("all_zero", "below_80")][:args.max_scenes]
    output = {"kind": "tto_controlled_score_diagnostics", "audit_sha256": digest(args.audit),
              "source_code_sha256": digest(source.__file__), "constructor_sha256": digest(constructor.__file__),
              "query_budget": 5*len(records), "official_queries": 0, "cases": [],
              "scope": "same original anchor; five fixed contrasts; no new-goal search or dataset admission",
              "trainable": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for record in records:
        case = {"token": record["token"], "split": record["split"], "category": record["category"],
                "results": {}, "status": "running"}
        try:
            directory = Path(audit["bank_root"]) / "scenes" / record["split"] / record["token"]
            if digest(directory / "candidates.npy") != record["candidate_sha256"]:
                raise ValueError("Candidate bank changed")
            contract = json.loads((directory / "contract.json").read_text())
            if digest(directory / "contract.json") != record["contract_sha256"]:
                raise ValueError("Contract changed")
            identity = contract["identity"]
            if identity["pdm_config"] != asdict(PDMScorerConfig()):
                raise ValueError("Official score configuration differs")
            with lzma.open(identity["official_cache_path"], "rb") as stream:
                cache = pickle.load(stream)
            pose = [cache.ego_state.rear_axle.x, cache.ego_state.rear_axle.y, cache.ego_state.rear_axle.heading]
            if (cache.ego_state.time_point.time_us != identity["timestamp_us"] or
                    not np.allclose(pose, identity["ego_pose"], atol=1e-9, rtol=0)):
                raise ValueError("Official cache anchor mismatch")
            rows = np.load(directory / "candidates.npy", allow_pickle=False)
            gt = np.asarray(identity["expert_gt"], dtype=np.float64)
            zero, _ = construct(gt, np.zeros((1, 2), dtype=np.int32), contract["protocol"])
            corrected = zero[0].copy()
            corrected[:, 2] = gt[:, 2].astype(np.float32)
            plans = {"gt_exact": gt, "stored_best": rows["trajectories"][record["best_row"]],
                     "zero_offset_warp": zero[0], "zero_offset_gt_heading": corrected,
                     "nearest_stored": rows["trajectories"][record["nearest_zero_row"]]}
            evaluator = Evaluator(cache)
            for name, plan in plans.items():
                output["official_queries"] += 1
                labels = evaluator.score(plan)
                case["results"][name] = {"labels": labels.tolist(), "trajectory": plan.tolist()}
            case["saved_gt_max_abs_difference"] = float(np.max(np.abs(
                np.asarray(case["results"]["gt_exact"]["labels"])-record["expert_labels"])))
            case["saved_best_max_abs_difference"] = float(np.max(np.abs(
                np.asarray(case["results"]["stored_best"]["labels"])-record["best_labels"])))
            case["parity_passed"] = max(case["saved_gt_max_abs_difference"], case["saved_best_max_abs_difference"]) < 1e-9
            case["status"] = "verified" if case["parity_passed"] else "score_context_mismatch"
        except Exception as error:
            case.update(status="unavailable", error=type(error).__name__+": "+str(error))
        output["cases"].append(case)
        output["wall_seconds"] = time.monotonic()-start
        pending = args.output.with_suffix(".partial")
        pending.write_text(json.dumps(output, indent=2, allow_nan=False))
        pending.replace(args.output)
        print(json.dumps({"token": case["token"], "status": case["status"],
                          "scores": {k:v["labels"][-1] for k,v in case["results"].items()},
                          "error": case.get("error")}), flush=True)


if __name__ == "__main__":
    main()
