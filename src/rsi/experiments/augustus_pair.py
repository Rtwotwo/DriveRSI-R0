"""Evidence-backed AUGUstus paired outcomes; no student training-source scoring.

This module records measured experiments. It cannot award a capability skill
credit when the executed branch was not selected by that skill beforehand.
"""

import copy
import json
import math
from pathlib import Path

import yaml

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rsi.decisions.capability_skills import SkillMemory
from driversi.training.orchestration.lifecycle import training_lineage


def _recipe(path):
    value = yaml.safe_load(Path(path).read_text())
    value = copy.deepcopy(value)
    value["training"].pop("output_dir", None)
    value["data"].pop("train_data_path", None)
    return value


def _positive_exposure(events):
    """Keep identities attached to the optimizer event that consumed them.

    A forward at zero LR can alter optimizer moments, but is not sufficient
    evidence for our stronger claim of direct synthetic parameter updates.
    """
    positive = set()
    for event in events:
        rates = event["learning_rates"]
        if not rates or any(
            type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in rates
        ):
            raise ValueError("Finite nonnegative effective learning rates required")
        if any(x > 0 for x in rates):
            positive.update(event["sample_ids"])
    return positive


def _paired_population(evaluated, subset):
    expected = {r["token"]: r["source_log"] for r in subset["records"]}
    rows = evaluated["paired_samples"]
    by_id = {r["sample_id"]: r for r in rows}
    if (
        not rows
        or len(by_id) != len(rows)
        or set(by_id) != set(expected)
        or any(r["source_log"] != expected[t] for t, r in by_id.items())
    ):
        raise ValueError("Evaluation rows must uniquely cover the locked population")
    for who in ("parent", "candidate"):
        for metric in ("PDMS", "NC", "DAC", "DDC", "EP", "TTC", "C"):
            values = [r[who][metric] for r in rows]
            declared = evaluated["metrics"][who][metric]
            if any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                for v in values + [declared]
            ):
                raise ValueError("Finite native PDM evidence in [0,1] required")
            # Floating summation tolerance; never alters any metric or gate.
            if not math.isclose(
                math.fsum(values) / len(values), declared, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("Aggregate PDM metric differs from actual paired rows")
    return by_id


def _fixed_cache_evidence(plans, expected):
    if not expected or any(any(p["files"].get(k) != v for k, v in expected.items()) for p in plans):
        raise ValueError("Executed original cache bytes differ between paired arms")


def _fixed_original_pool(control_plan, joint_plan, joint):
    # The treatment plan seals the materialized mixed Arrow cache, not the
    # original train directory. Its immutable joint manifest supplies that
    # original-cache closure. Do not rewrite a historical plan to add hashes.
    manifest_path = joint_plan["data"]["manifest"]
    if joint_plan["files"].get(manifest_path) != digest(manifest_path):
        raise ValueError("Executed joint manifest changed")
    _fixed_cache_evidence([control_plan], joint["base_cache_files"])
    _fixed_cache_evidence([joint_plan], joint["train_cache_files"])


def _evaluation_attempt(root, report):
    """A retry may replace only evaluation, with unchanged learned ancestry.

    Keep old attempt files intact. The new worker must consume the exact resume
    receipt and checkpoint, not another arm or a newly trained replacement.
    """
    path = Path(report).resolve()
    plan = read_mapping(path.parent / "training-plan.json")
    ancestor = (Path(root) / "resume/training-result.json").resolve()
    if (
        path.name != "training-result.json"
        or plan.get("phase") != "evaluate"
        or Path(plan["output"]).resolve() != path.parent
        or Path(plan["previous"]).resolve() != ancestor
        or plan["files"].get(str(ancestor)) != digest(ancestor)
    ):
        raise ValueError("Reevaluation must seal the same learned ancestor")
    learned = read_mapping(ancestor)
    if any(plan["files"].get(k) != sha for k, sha in learned["checkpoint_files"].items()):
        raise ValueError("Reevaluation checkpoint bytes differ")
    for name, sha in plan["files"].items():
        if digest(name) != sha:
            raise ValueError("Reevaluation input changed: " + name)
    value = read_mapping(path)
    worker = read_mapping(path.parent / "worker-result.json")
    for key in ("status", "metrics", "metric_scope", "paired_samples", "evaluated_checkpoint"):
        if value.get(key) != worker.get(key):
            raise ValueError("Reevaluation differs from actual worker evidence")
    return path


def _arm(root, evaluation_report=None):
    root = Path(root).resolve()
    learned = read_mapping(root / "resume/training-result.json")
    evaluation_path = (
        _evaluation_attempt(root, evaluation_report)
        if evaluation_report
        else root / "evaluate/training-result.json"
    )
    evaluated = read_mapping(evaluation_path)
    plan = read_mapping(root / "train/training-plan.json")
    if any(
        r.get("model") != "augustus" or r.get("status") != "completed" for r in (learned, evaluated)
    ):
        raise ValueError("Completed native AUGUstus learning and evaluation required")
    if (
        learned["context"] != evaluated["context"]
        or evaluated["evaluated_checkpoint"] != learned["checkpoint"]
        or evaluated.get("metric_scope")
        not in (
            "native_navsim_pdm_on_locked_official_navtest_pilot",
            "native_navsim_pdm_on_full_official_navtest",
        )
    ):
        raise ValueError("Actual checkpoint and locked official evaluation must agree")
    lineage = training_lineage(root / "resume/training-result.json")
    seen, rates, receipts, all_events = [], [], {}, []
    for phase in ("train", "resume"):
        report = read_mapping(root / phase / "training-result.json")
        journal = root / phase / "consumed-training-samples.jsonl"
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        if len(events) != report["updates"] or any(not r["sample_ids"] for r in events):
            raise ValueError("Every optimizer update must have actual forward identities")
        seen.extend(t for row in events for t in row["sample_ids"])
        rates.extend(row["learning_rates"] for row in events)
        all_events.extend(events)
        for name in (
            "training-result.json",
            "training-plan.json",
            "native.yaml",
            "consumed-training-samples.jsonl",
        ):
            p = root / phase / name
            receipts[str(p)] = digest(p)
    for name in ("training-result.json", "training-plan.json"):
        p = evaluation_path.parent / name
        receipts[str(p)] = digest(p)
    if evaluation_report:
        for name in (
            "worker-result.json",
            "parent-pdm-inputs.npz",
            "candidate-pdm-inputs.npz",
            "parent-pdm-samples.json",
            "candidate-pdm-samples.json",
        ):
            p = evaluation_path.parent / name
            receipts[str(p)] = digest(p)
        old = root / "evaluate/training-result.json"
        receipts[str(old)] = digest(old)
    return dict(
        root=root,
        learned=learned,
        evaluated=evaluated,
        plan=plan,
        seen=seen,
        rates=rates,
        receipts=receipts,
        lineage=lineage,
        events=all_events,
        positive_seen=_positive_exposure(all_events),
        evaluation_plan=read_mapping(evaluation_path.parent / "training-plan.json"),
    )


def _same_parent_predictions(left, right):
    """Identical aggregate scores cannot conceal different parent trajectories."""
    import numpy as np

    arrays = []
    for path in (left, right):
        with np.load(path, allow_pickle=False) as data:
            tokens = data["tokens"].tolist()
            pred = data["trajectories"]
            if (
                len(tokens) != len(set(tokens))
                or len(tokens) != len(pred)
                or not np.isfinite(pred).all()
            ):
                raise ValueError("Invalid parent prediction population")
            arrays.append(dict(zip(tokens, pred, strict=True)))
    if arrays[0].keys() != arrays[1].keys() or any(
        not np.array_equal(arrays[0][key], arrays[1][key]) for key in arrays[0]
    ):
        raise ValueError("Parent prediction arrays differ between paired evaluations")


def compare(
    control, treatment, *, generation_costs=None, evaluation_reports=None, relabel_report=None
):
    """Recompute paired outcomes from actual workers and admitted joint caches."""
    from driversi_agents.augustus.branch_supervision import validate_sample
    from driversi_agents.augustus.data import manifest

    if evaluation_reports is not None and set(evaluation_reports) != {"control", "treatment"}:
        raise ValueError("Both arms require explicit reevaluation receipts")
    reports = evaluation_reports or {}
    left, right = (_arm(control, reports.get("control")), _arm(treatment, reports.get("treatment")))
    lp, rp = left["plan"], right["plan"]
    for key in ("pilot", "budget", "mode", "initialization"):
        if lp["resolved"]["request"][key] != rp["resolved"]["request"][key]:
            raise ValueError("Paired training budgets or initialization differ")
    if (
        left["learned"]["total_updates"] != right["learned"]["total_updates"]
        or left["rates"] != right["rates"]
        or _recipe(left["root"] / "train/native.yaml")
        != _recipe(right["root"] / "train/native.yaml")
        or _recipe(left["root"] / "resume/native.yaml")
        != _recipe(right["root"] / "resume/native.yaml")
    ):
        raise ValueError("Effective native recipe, rates or update counts differ")
    parents = [{k: v for k, v in p["files"].items() if "/weights/augustus/" in k} for p in (lp, rp)]
    if not parents[0] or parents[0] != parents[1]:
        raise ValueError("Exact initialized parent assets differ")
    cm = manifest(lp["data"]["manifest"], allow_smoke=True)
    tm = manifest(rp["data"]["manifest"], allow_smoke=True)
    joint = read_mapping(rp["data"]["manifest"])
    if relabel_report is None:
        if joint.get("kind") != "augustus_joint_cache":
            raise ValueError("Actual admitted joint supervision required")
        _fixed_original_pool(lp, rp, joint)
        base = manifest(joint["base_manifest"]["path"], allow_smoke=True)
        if cm != base:
            raise ValueError("Control is not exactly the original intervention pool")
        samples = [validate_sample(x["path"]) for x in joint["synthetic_samples"]]
        added = {s["sample_id"] for s in samples}
    else:
        from driversi_agents.augustus.data import read_cache

        proof = read_mapping(relabel_report)
        if (
            proof.get("kind") != "augustus_relabel_dataset"
            or proof["label_source"] != "drivetto"
            or proof["manifest_sha256"] != digest(rp["data"]["manifest"])
            or proof["source_manifest_sha256"] != digest(lp["data"]["manifest"])
        ):
            raise ValueError("Actual same-anchor relabel provenance required")
        _fixed_cache_evidence([lp], proof["source_cache_files"])
        _fixed_cache_evidence([rp], proof["cache_files"])
        base = cm
        samples = proof["replacements"]
        added = {x["token"] for x in samples}
        original = {row["frame_id"]: row for row in read_cache(cm["splits"]["train"])}
        changed = {row["frame_id"]: row for row in read_cache(tm["splits"]["train"])}
        if original.keys() != changed.keys() or not added <= original.keys():
            raise ValueError("Relabel must preserve the complete original training pool")
        for token, row in original.items():
            other = changed[token]
            fields = (
                {"future_x_y_angle", "solution_planning", "reasoning_trace"}
                if token in added
                else set()
            )
            if row.keys() != other.keys() or any(
                row[k] != other[k] for k in row if k not in fields
            ):
                raise ValueError("Relabel changed original observations or prediction supervision")
            if token in added and row["future_x_y_angle"] == other["future_x_y_angle"]:
                raise ValueError("Augmentation label is unchanged")
    if not added or not added <= right["positive_seen"]:
        raise ValueError("Every admitted augmented sample needs an actual positive-LR update")
    for arm, data in ((left, cm), (right, tm)):
        allowed = {r["frame_id"] for r in data["splits"]["train"]["records"]}
        if not set(arm["seen"]) <= allowed:
            raise ValueError("Training consumed undeclared sample identities")
    evaluation_plans = [a["evaluation_plan"] for a in (left, right)]
    _same_parent_predictions(
        *(Path(p["output"]) / "parent-pdm-inputs.npz" for p in evaluation_plans)
    )
    for key in ("evaluation_protocol", "evaluation_subset"):
        if evaluation_plans[0]["data"][key] != evaluation_plans[1]["data"][key]:
            raise ValueError("Locked evaluation inputs differ")
    evaluator_values = [p["evaluation_context"]["value"] for p in evaluation_plans]
    for key in ("model", "protocol", "development_source_logs", "metric_cache", "files"):
        if evaluator_values[0][key] != evaluator_values[1][key]:
            raise ValueError("Actually executed evaluation protocol/cache differs")
    metric_cache = evaluator_values[0]["metric_cache"]
    subset_path = evaluation_plans[0]["data"]["evaluation_subset"]
    subset = read_mapping(subset_path)
    eval_files = {
        k: v
        for k, v in evaluation_plans[0]["files"].items()
        if k == subset_path or k.startswith(metric_cache + "/")
    }
    _fixed_cache_evidence(evaluation_plans, eval_files)
    if eval_files.get(subset_path) != digest(subset_path):
        raise ValueError("Locked population changed after execution")
    pairs = [_paired_population(a["evaluated"], subset) for a in (left, right)]
    if not pairs[0] or pairs[0].keys() != pairs[1].keys():
        raise ValueError("Paired evaluation populations differ")
    regressions = []
    for token, x in pairs[0].items():
        y = pairs[1][token]
        if x["source_log"] != y["source_log"] or x["parent"] != y["parent"]:
            raise ValueError("Parent predictions or test-source identity differ")
        for metric in ("NC", "DAC", "DDC", "TTC"):
            if y["candidate"][metric] < x["candidate"][metric]:
                # Internal evidence only; public reports must aggregate these.
                regressions.append(dict(sample_id=token, metric=metric))
    metrics = {
        "parent": left["evaluated"]["metrics"]["parent"],
        "control": left["evaluated"]["metrics"]["candidate"],
        "treatment": right["evaluated"]["metrics"]["candidate"],
    }
    gain = 100 * (metrics["treatment"]["PDMS"] - metrics["control"]["PDMS"])
    costs = {
        name: {"training": arm["lineage"]["cost"], "evaluation": arm["evaluated"]["consumed"]}
        for name, arm in (("control", left), ("treatment", right))
    }
    # Rendering/annotation/search costs cannot be inferred from training receipts.
    return dict(
        schema_version=1,
        kind="augustus_paired_feedback",
        status="completed",
        metrics=metrics,
        gain_vs_control_pdms_points=gain,
        gain_vs_parent_pdms_points=100 * (metrics["treatment"]["PDMS"] - metrics["parent"]["PDMS"]),
        safety_regressions=regressions,
        effective_recipe_verified=True,
        exposed_added_sample_ids=sorted(added),
        original_rows=len(base["splits"]["train"]["records"]),
        positive_lr_added_sample_ids=sorted(added & right["positive_seen"]),
        synthetic_rows=len(samples),
        evaluation_samples=len(pairs[0]),
        augmentation_kind="original_anchor_relabel" if relabel_report else "reactive_new_history",
        new_images=0 if relabel_report else None,
        evaluation_scope=(
            "full_official_navtest_adaptive"
            if evaluation_plans[0]["data"]["evaluation_protocol"] == "official_navtest_v1"
            else "locked_official_navtest_pilot_adaptive_not_full_benchmark"
        ),
        parent_assets=parents[0],
        training_evaluation_costs=costs,
        generation_costs=generation_costs,
        complete_total_cost=False,
        evidence_files={**left["receipts"], **right["receipts"]},
        promoted=False,
        reliable_skill_improvement=False,
        limitation="Pilot gain is not causal evidence of memory-policy superiority",
    )


def record(report, output, memory, *, resume=False):
    """Store a sealed measured learning observation; no retroactive skill credit."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not resume or read_mapping(output) != report:
            raise ValueError("Preserve previous feedback; choose a new output")
    else:
        output.write_text(json.dumps(report, indent=2, allow_nan=False))
    observation = dict(
        evidence_id=digest(output),
        outcome_type="paired_training_observation",
        context={
            "model": "augustus",
            "parent_assets": report["parent_assets"],
            "evaluator_version": report["evaluation_scope"],
        },
        capability=None,
        skill=(
            "same_anchor_relabel"
            if report.get("augmentation_kind") == "original_anchor_relabel"
            else "reactive_new_history"
        ),
        skill_version=1,
        credited=False,
        reasons=["no_executed_capability_plan_binding"],
        gain_vs_control=report["gain_vs_control_pdms_points"],
        safety_passed=not report["safety_regressions"],
        cost_wall_seconds=None,
        measured_partial_cost=report["training_evaluation_costs"],
        report=str(output),
        report_sha256=digest(output),
        stable_improvement_demonstrated=False,
    )
    store = SkillMemory(memory)
    try:
        return store.record(observation)
    finally:
        store.close()
