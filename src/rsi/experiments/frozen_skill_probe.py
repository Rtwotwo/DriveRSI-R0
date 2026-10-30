"""DriveHarness pilot for frozen AUGUstus prompt skills on real log anchors."""

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.runtime.budget import write_json
from driversi.rsi.decisions.frozen_skills import ARMS, analyze_pairs, choose_skill


def read(path):
    return json.loads(Path(path).read_text())


def prepare(config, output):
    """Freeze membership and source identity before any new model inference."""
    c = read(config)
    out = Path(output).resolve()
    root = project_root()
    if not out.is_relative_to(root / "exp"):
        raise ValueError("Experiment artifacts belong under exp")
    if c["count"] != 32 or c["seeds"] != [42, 43]:
        raise ValueError("This pilot declares 32 independent logs and two paired seeds")
    prior = read(c["baseline_report"])
    prior_plan = read(c["baseline_plan"])
    if not prior["complete"] or len(prior["rows"]) != 12146:
        raise ValueError("Complete same-model baseline required for retrospective selection")
    if prior["plan_sha256"] != digest(c["baseline_plan"]):
        raise ValueError("Historical parent report and plan differ")
    checkpoint = Path(c["checkpoint"])
    checkpoint_sha = digest(checkpoint / "model.safetensors")
    historical_sha = prior_plan["inputs"].get(str(checkpoint / "model.safetensors"))
    if historical_sha != checkpoint_sha:
        raise ValueError("Selection must use this exact frozen AUGUstus parent")
    selected, logs = [], set()
    ordered = sorted(prior["rows"], key=lambda r: hashlib.sha256(("frozen-prefix-v1/"+r["token"]).encode()).hexdigest())
    strata = [("previous_zero", 12, lambda x: x == 0),
              ("previous_low", 12, lambda x: 0 < x < 0.6),
              ("previous_other", 8, lambda x: x >= 0.6)]
    for name, count, accepts in strata:
        group = []
        for row in ordered:
            if row["source_log"] in logs or not accepts(row["metrics"]["PDMS"]): continue
            group.append(dict(token=row["token"], source_log=row["source_log"], selection_stratum=name))
            logs.add(row["source_log"])
            if len(group) == count: break
        if len(group) != count: raise ValueError("Insufficient distinct logs for declared strata")
        selected.extend(group)
    lookup = {r["token"]: i for i, rows in enumerate(prior_plan["chunks"]) for r in rows}
    original_root = Path(c["baseline_plan"]).parent
    source_receipts = {}
    for member in selected:
        chunk = original_root / "chunks" / f"{lookup[member['token']]:04d}"
        cache_receipt = chunk / "data/receipt.json"
        completed = read(chunk / "completed.json")
        score_report = read(completed["report"])
        if digest(completed["report"]) != completed["sha256"]:
            raise ValueError("Historical score report changed")
        native = read(cache_receipt)
        meta = next(r for r in native["block"]["records"] if r["frame_id"] == member["token"])
        if meta["log_name"] != member["source_log"]: raise ValueError("Cache lineage mismatch")
        source_receipts[str(cache_receipt)] = digest(cache_receipt)
        files = score_report["inputs"]["metric_files"]
        metric = next(p for p in files if Path(p).parent.name == member["token"])
        member.update(cache_receipt=str(cache_receipt), metric_source=metric,
                      metric_sha256=files[metric])
    sources = [config, c["baseline_report"], c["baseline_plan"], c["skills"],
               root / "src/rsi/decisions/frozen_skills.py",
               root / "src/rsi/experiments/frozen_skill_probe.py",
               root / "agents/augustus/skill_inference.py",
               root / "src/rsi/plugins/frozen_skills.py"]
    # Existing model identity convention: freeze full parent bundle plus native action path.
    sources += [p for p in checkpoint.iterdir() if p.is_file()]
    sources += [root / "agents/augustus/native/AUGUstus_SFT" / name for name in
                ["model/model_qwen3p5.py", "model/dataset.py", "model/utils.py", "test_qwen3p5.py",
                 "train_qwen3p5.py", "prompts/default.txt", "configs/navsim_qwen3p5.yaml"]]
    plan = dict(schema_version=1, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                members=selected, seeds=c["seeds"], arms=list(ARMS), skills=read(c["skills"]),
                files={str(p): digest(p) for p in sources}, source_receipts=source_receipts,
                expected_predictions=32*2*3, scoring_agent_steps=32*2*3*80,
                no_optimizer=True, scope="retrospective_navtest_development", gpu=c["gpu"])
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError("Existing pilot plan differs; do not overwrite")
    write_json(plan_path, plan)
    task = dict(schema_version=1, id=c["task_id"], label="AUGUstus frozen prefix skill pilot",
                budget=dict(max_wall_seconds=7200, max_sim_agent_steps=plan["scoring_agent_steps"]), steps=[])
    for name, provider, inputs, cap in [
        ("prepare", "frozen_skill_observations", {}, 1200),
        ("inference", "native_model_frozen_skill_inference", {"observations":"prepare"}, 3600),
        ("scoring", "frozen_skill_native_scoring", {"observations":"prepare", "predictions":"inference"}, 1200),
        ("analysis", "frozen_skill_paired_analysis", {"observations":"prepare", "scores":"scoring"}, 300)]:
        task["steps"].append(dict(id=name, provider=provider, inputs=inputs,
            config=dict(plan=str(plan_path)), max_wall_seconds=cap))
    write_json(out / "task.json", task)
    return plan


def validate_plan(path):
    p = read(path)
    for name, sha in {**p["files"], **p["source_receipts"]}.items():
        if digest(name) != sha: raise ValueError("Frozen pilot input changed: " + name)
    return p


def prepare_data(plan_path, output):
    from datasets import concatenate_datasets
    from driversi_agents.augustus.data import read_cache

    p = validate_plan(plan_path)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    parts, members, loaded, destinations = [], [], {}, []
    for member in p["members"]:
        receipt = member["cache_receipt"]
        if receipt not in loaded:
            data = read(receipt)
            for f, sha in data["files"].items():
                if digest(f) != sha: raise ValueError("Original native cache changed")
            ds = read_cache(data["block"])
            loaded[receipt] = (ds, {v: i for i, v in enumerate(ds["frame_id"])})
        ds, indices = loaded[receipt]
        idx = indices[member["token"]]; row = ds[idx]
        if row["scene_token"] != member["source_log"]: raise ValueError("Row belongs to another log")
        velocity = {}
        for key in ("vx", "vy"):
            match = re.search(r"\b"+key+r":\s*([-+0-9.eE]+)\s*m/s", row["ego_dynamic_state"])
            if not match: raise ValueError("Cannot parse native current velocity")
            velocity[key] = float(match.group(1))
        observation = dict(navigation_command=row["driving_command"],
                           speed_mps=math.hypot(velocity["vx"], velocity["vy"]))
        skill = choose_skill(observation, p["skills"])
        members.append({**member, "skill":skill})
        parts.append(ds.select([idx]))
        source = Path(member["metric_source"])
        if digest(source) != member["metric_sha256"]: raise ValueError("Native metric context changed")
        target = output / "metric_cache" / member["source_log"] / "unknown" / member["token"] / "metric_cache.pkl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        destinations.append(str(target))
    cache = output / "cache"
    concatenate_datasets(parts).save_to_disk(str(cache / "batch_0"))
    metadata = output / "metric_cache/metadata/metric_cache_metadata_node_0.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text("file_name\n" + "\n".join(destinations) + "\n")
    report = dict(plan=p, plan_sha256=digest(plan_path), members=members, cache=str(cache),
                  metric_cache=str(output / "metric_cache"),
                  cache_files={str(f):digest(f) for f in cache.rglob('*') if f.is_file()},
                  metric_files={str(f):digest(f) for f in [metadata, *map(Path,destinations)]},
                  future_targets_not_model_inputs=True, optimizer_updates=0)
    write_json(output / "report.json", report)
    return report


def score(prepared_path, predictions_path, inference_report_path, output):
    import numpy as np
    prepared = read(prepared_path); inference = read(inference_report_path)
    if inference["prepared_sha256"] != digest(prepared_path): raise ValueError("Prediction lineage mismatch")
    if inference["files"].get(str(predictions_path)) != digest(predictions_path):
        raise ValueError("Unsealed or modified trajectories")
    for f, sha in prepared["metric_files"].items():
        if digest(f) != sha: raise ValueError("Scoring context changed")
    rows = read(predictions_path); expected = {(m["token"], s, a) for m in prepared["members"]
                                               for s in prepared["plan"]["seeds"] for a in prepared['plan'].get('arms', ARMS)}
    keys = [(r["token"], r["seed"], r["arm"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete sealed inference coverage")
    sft = project_root() / "agents/augustus/native/AUGUstus_SFT"
    sys.path[:0] = [str(sft), str(sft/"third_party/navsim"), str(sft/"third_party/nuplan-devkit")]
    from model.utils import compute_pdm_scores_batch
    from driversi_agents.augustus.budget import metered
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    scorer = metered(compute_pdm_scores_batch, output/"scoring-budget.json", len(rows)*80)
    metrics = scorer(np.asarray([r["trajectory"] for r in rows], dtype=np.float32),
                     [r["token"] for r in rows], num_workers=8,
                     metric_cache_path=prepared["metric_cache"], return_per_sample=True,
                     return_count=True)
    if metrics.get("count") != len(rows) or any(v is None for v in metrics["per_sample"]):
        raise RuntimeError("Native scoring did not cover every prediction")
    scored = [{**r, "metrics":m} for r,m in zip(rows, metrics.pop("per_sample"), strict=True)]
    write_json(output / "scores.json", scored)
    report = dict(status="completed", completed=len(scored), source_predictions_sha256=digest(predictions_path),
                  prepared_sha256=digest(prepared_path), scores_sha256=digest(output/"scores.json"),
                  wall_seconds=time.monotonic()-started, sim_agent_steps=len(rows)*80,
                  protocol="unchanged_native_augustus_pdm_v1", scores=str(output/"scores.json"))
    write_json(output/"report.json", report)
    return report


def analyze(prepared_path, score_report_path, output):
    p = read(prepared_path); scoring = read(score_report_path)
    if scoring["prepared_sha256"] != digest(prepared_path) or digest(scoring["scores"]) != scoring["scores_sha256"]:
        raise ValueError("Scored evidence lineage mismatch")
    expected = {(m["token"], s, a) for m in p["members"] for s in p["plan"]["seeds"] for a in ARMS}
    report = analyze_pairs(read(scoring["scores"]), expected)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    report.update(unique_logs=len({m['source_log'] for m in p['members']}),
                  native_score_scale="0_to_1", memory_update="candidate_evidence_only",
                  checkpoint_sha256=p['plan']['checkpoint_sha256'])
    write_json(output/"report.json", report)
    write_json(output/"skill-experience.json", dict(version=p['plan']['skills']['version'],
        disposition=report['disposition'], default_policy="native", promoted=False,
        reason="Requires independent cross-log validation before skill promotion",
        report=str(output/"report.json"), report_sha256=digest(output/"report.json")))
    lines = ["# AUGUstus frozen skill pilot", "", "Retrospective development evidence; no training or skill promotion.", "",
             "| Arm | PDMS /100 | Zero pairs | Mean inference seconds |", "|---|---:|---:|---:|"]
    for arm, row in report['arms'].items():
        lines.append(f"| {arm} | {row['metrics']['PDMS']*100:.4f} | {row['zero_count']} | {row['mean_inference_seconds']:.3f} |")
    lines += ["", "Disposition: " + report['disposition'], "",
              "32 distinct logs, two paired seeds. A trajectory change is not a driving gain. Full test generalization and recursive improvement are not demonstrated."]
    (output/"report.md").write_text('\n'.join(lines)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config'); parser.add_argument('output')
    args=parser.parse_args()
    p=prepare(args.config,args.output)
    print(json.dumps(dict(prepared=True, members=len(p['members']), predictions=p['expected_predictions'])))


if __name__ == '__main__': main()
