"""Prospective, resumable supervision of existing native AUGUstus paired skills.

A campaign executes only sealed service programs; proposals cannot change their
native model, loss, admission or evaluator. Runtime VLA intervention is not an
available skill. All raw evidence stays in exp, never in the public workbench.
"""

import fcntl
from pathlib import Path

from driversi.configuration import read_mapping
from driversi.data.catalog.store import identity
from driversi.rsi.decisions.intervention_policy import select_intervention
from driversi.rsi.decisions.service_memory import review_learning
from driversi.rsi.experiments import augustus_round
from driversi.runtime.budget import write_json


def _normal_program(path):
    plan = read_mapping(path)
    if plan["program_id"] != identity({k: v for k, v in plan.items() if k != "program_id"}):
        raise ValueError("Original program identity changed")
    augustus_round._verify(plan["files"])
    augustus_round._verify(plan["code_files"])
    return plan


def _canonical_assets(assets):
    import os

    if not assets:
        raise ValueError("Complete parent assets required")
    common = Path(os.path.commonpath(list(assets)))
    if common.is_file():
        common = common.parent
    return {str(Path(p).relative_to(common)): sha for p, sha in assets.items()}


def _verify_parent_assets(plan, actual):
    from driversi.agents.backends.native import path_at, project_root
    from driversi_agents.augustus.adapter import complete_assets

    model = read_mapping(plan["model_template"])
    parent = path_at(project_root(), model["weights"]["parent"])
    expected = {str(p): plan["files"].get(str(p)) for p in complete_assets(parent)}
    if None in expected.values() or identity(expected) != plan["context"]["parent_assets_id"]:
        raise ValueError("Parent assets differ from original context")
    if _canonical_assets(expected) != _canonical_assets(actual):
        raise ValueError("Executed parent contents differ from sealed original")


def _comparison_key(plan, assets=None):
    """Relocated identical weights are comparable; changed native scorers are not."""
    from driversi.agents.backends.native import path_at, project_root
    from driversi_agents.augustus.adapter import complete_assets

    if assets is None:
        model = read_mapping(plan["model_template"])
        parent = path_at(project_root(), model["weights"]["parent"])
        assets = augustus_round._files(complete_assets(parent))
    augustus_round._verify(assets)
    relative = _canonical_assets(assets)
    marker = "/agents/augustus/native/AUGUstus_SFT/third_party/"
    scorers = {p.split(marker, 1)[1]: sha for p, sha in plan["code_files"].items() if marker in p}
    if not relative or not scorers:
        raise ValueError("Complete parent and native scorer identities required")
    return (
        plan["context"]["model"],
        identity(relative),
        plan["context"]["evaluator_id"],
        identity(scorers),
    )


def checked_review(path):
    """Bind history to executed programs and recompute scores from native receipts."""
    from driversi.rsi.experiments import augustus_pair

    path = Path(path).resolve(strict=True)
    value = read_mapping(path)
    if value.get("promoted") is not False or value.get("skill_gain_credited") is not False:
        raise ValueError("Pilot observations cannot establish model/skill promotion")
    plan = read_mapping(path.parent / "program.json")
    if value.get("context") != plan["context"] or value.get("program_id") != plan["program_id"]:
        raise ValueError("Review context/program differs from executed program")
    original = plan
    evaluation_reports = None
    if plan.get("continuation_kind"):
        if plan["continuation_kind"] != "evaluation_only_no_optimizer_updates":
            raise ValueError("Unsupported historical continuation")
        original = _normal_program(Path(plan["parent_program"]) / "program.json")
        files = plan["continuation_files"]
        augustus_round._verify(files)
        if (
            plan["context"] != original["context"]
            or plan["parent_program_id"] != original["program_id"]
            or plan["program_id"]
            != identity(
                dict(kind="evaluation_recovery", parent=original["program_id"], files=files)
            )
        ):
            raise ValueError("Historical evaluation continuation differs from parent")
        executions = [read_mapping(p) for p in files if Path(p).name == "execution.json"]
        if (
            len(executions) != 1
            or executions[0]["optimizer_updates"] != 0
            or executions[0]["parent_program_id"] != original["program_id"]
        ):
            raise ValueError("Sealed evaluation-only recovery required")
        evaluation_reports = {
            arm: executions[0]["rows"][arm]["report"] for arm in ("control", "treatment")
        }
        for arm, report in evaluation_reports.items():
            if files.get(report) != executions[0]["rows"][arm]["sha256"]:
                raise ValueError("Recovery receipt does not seal actual evaluation")
    else:
        _normal_program(path.parent / "program.json")
    evidence = value.get("execution_evidence", {})
    if not evidence:
        raise ValueError("Sealed execution evidence required")
    augustus_round._verify(evidence)
    if value["node"] != "learning_review":
        raise ValueError("Historical seed must have completed paired learning review")
    refs = value.get("paired_feedback", {})
    if len(refs) != 1:
        raise ValueError("One sealed paired feedback required")
    augustus_round._verify(refs)
    feedback = read_mapping(next(iter(refs)))
    augustus_round._verify(feedback["evidence_files"])
    roots = [
        Path(original["output"]) / "tasks" / (original["id"] + "-" + arm)
        for arm in ("control", "treatment")
    ]
    arms = [
        augustus_pair._arm(root, (evaluation_reports or {}).get(arm))
        for root, arm in zip(roots, ("control", "treatment"))
    ]
    checks = [
        read_mapping(root / "feedback/feedback.json")["cross_process_resume"] for root in roots
    ]
    if checks != value["resume_checks"]:
        raise ValueError("Recovery claims differ from actual feedback receipts")
    for key in ("pilot", "budget", "mode", "initialization"):
        if (
            arms[0]["plan"]["resolved"]["request"][key]
            != arms[1]["plan"]["resolved"]["request"][key]
        ):
            raise ValueError("Historical paired training recipes differ")
    if (
        arms[0]["learned"]["total_updates"] != arms[1]["learned"]["total_updates"]
        or arms[0]["rates"] != arms[1]["rates"]
        or not set(feedback["positive_lr_added_sample_ids"]) <= arms[1]["positive_seen"]
    ):
        raise ValueError("Historical updates or synthetic exposure differ")
    for phase in ("train", "resume"):
        if augustus_pair._recipe(roots[0] / phase / "native.yaml") != augustus_pair._recipe(
            roots[1] / phase / "native.yaml"
        ):
            raise ValueError("Historical effective native recipes differ")
    subset_path = arms[0]["evaluation_plan"]["data"]["evaluation_subset"]
    if any(arm["evaluation_plan"]["data"]["evaluation_subset"] != subset_path for arm in arms):
        raise ValueError("Historical evaluation populations differ")
    subset = read_mapping(subset_path)
    pairs = [augustus_pair._paired_population(arm["evaluated"], subset) for arm in arms]
    augustus_pair._same_parent_predictions(
        *(Path(arm["evaluation_plan"]["output"]) / "parent-pdm-inputs.npz" for arm in arms)
    )
    metrics = dict(
        parent=arms[0]["evaluated"]["metrics"]["parent"],
        control=arms[0]["evaluated"]["metrics"]["candidate"],
        treatment=arms[1]["evaluated"]["metrics"]["candidate"],
    )
    regressions = []
    for token, left in pairs[0].items():
        right = pairs[1][token]
        if left["parent"] != right["parent"]:
            raise ValueError("Historical parent metric rows differ")
        for metric in ("NC", "DAC", "DDC", "TTC"):
            if right["candidate"][metric] < left["candidate"][metric]:
                regressions.append(dict(sample_id=token, metric=metric))
    recomputed = dict(
        feedback,
        metrics=metrics,
        safety_regressions=regressions,
        gain_vs_control_pdms_points=100
        * (metrics["treatment"]["PDMS"] - metrics["control"]["PDMS"]),
        gain_vs_parent_pdms_points=100 * (metrics["treatment"]["PDMS"] - metrics["parent"]["PDMS"]),
    )
    for key in (
        "metrics",
        "safety_regressions",
        "gain_vs_control_pdms_points",
        "gain_vs_parent_pdms_points",
    ):
        if feedback[key] != recomputed[key]:
            raise ValueError("Saved feedback differs from actual worker results")
    _verify_parent_assets(original, feedback["parent_assets"])
    verified = review_learning(recomputed, checks)
    for key in ("node", "status", "observed_gain", "gain_vs_parent", "safety_passed"):
        if value.get(key) != verified.get(key):
            raise ValueError("Review differs from independently checked paired feedback")
    # Persist every dependency used above, including the frozen runtime's code.
    # A seed is cached evidence; advance must detect changes without recomputing it.
    dependencies = {
        **original["files"],
        **original["code_files"],
        **plan.get("continuation_files", {}),
        **evidence,
        **refs,
        **feedback["evidence_files"],
        **feedback["parent_assets"],
        **augustus_round._files([path, path.parent / "program.json", subset_path]),
    }
    if plan.get("parent_program"):
        dependencies.update(augustus_round._files([Path(plan["parent_program"]) / "program.json"]))
    for root, arm in zip(roots, arms):
        dependencies.update(arm["receipts"])
        dependencies.update(arm["evaluation_plan"]["files"])
        dependencies.update(
            augustus_round._files(
                [
                    root / "feedback/feedback.json",
                    Path(arm["evaluation_plan"]["output"]) / "parent-pdm-inputs.npz",
                ]
            )
        )
        dependencies.update({item["path"]: item["sha256"] for item in arm["lineage"]["receipts"]})
    augustus_round._verify(dependencies)
    return dict(
        value,
        comparison_key=_comparison_key(original, feedback["parent_assets"]),
        validated_dependency_files=dependencies,
    )


def prepare(config_path):
    """Seal every candidate and total update cap before choosing the first trial."""
    config_path = Path(config_path).resolve(strict=True)
    config = read_mapping(config_path)
    if set(config) != {
        "schema_version",
        "id",
        "output",
        "candidates",
        "seed_reviews",
        "max_trials",
        "max_optimizer_updates_including_replays",
    }:
        raise ValueError("Explicit finite intervention campaign required")
    if config["schema_version"] != 1 or type(config["max_trials"]) is not int:
        raise ValueError("Unsupported campaign schema")
    rows = config["candidates"]
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= 4
        or not 1 <= config["max_trials"] <= len(rows)
    ):
        raise ValueError("Pilot allows 1-4 candidate programs")
    cap = config["max_optimizer_updates_including_replays"]
    if type(cap) is not int or not 1 <= cap <= 192:
        raise ValueError("Bounded actual optimizer budget required, including replay")
    out = augustus_round._exp(config["output"])
    if out.exists():
        raise ValueError("Use a new campaign directory, or advance the existing one")
    programs, files, ids, contexts = [], augustus_round._files([config_path]), set(), set()
    for row in rows:
        if set(row) != {"id", "priority", "hypothesis", "program"}:
            raise ValueError("Only registered paired-service candidates are supported")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
            raise ValueError("Distinct intervention identities required")
        if (
            type(row["priority"]) is not int
            or not isinstance(row["hypothesis"], str)
            or not row["hypothesis"].strip()
        ):
            raise ValueError("Prospective priority and hypothesis required")
        ids.add(row["id"])
        directory = augustus_round._exp(row["program"])
        plan = augustus_round.verify_program(directory)
        if (directory / "review.json").exists() or (directory / "harness.sqlite").exists():
            raise ValueError("Candidate already executed: use it as seed evidence, not a new trial")
        contexts.add(_comparison_key(plan))
        req = read_mapping(plan["request_template"])
        # Two arms, each train + continuation + replay of continuation.
        updates = 6 * req["budget"]["max_optimizer_steps"]
        if any(p["program_id"] == plan["program_id"] for p in programs):
            raise ValueError("Duplicate program cannot be a distinct intervention")
        programs.append(
            {
                **row,
                "program": str(directory),
                "program_id": plan["program_id"],
                "optimizer_update_reservation": updates,
            }
        )
        files.update(augustus_round._files([directory / "program.json"]))
    if len(contexts) != 1:
        raise ValueError("Campaign candidates must share parent and locked evaluation")
    seed_paths = config["seed_reviews"]
    if (
        not isinstance(seed_paths, list)
        or len(seed_paths) > 8
        or len(set(seed_paths)) != len(seed_paths)
    ):
        raise ValueError("Bounded distinct seed observations required")
    history = []
    for p in seed_paths:
        value = checked_review(p)
        if tuple(value["comparison_key"]) not in contexts:
            raise ValueError("Historical review belongs to another parent/evaluation")
        history.append({k: value[k] for k in ("node", "status", "observed_gain")})
        files.update(augustus_round._files([p]))
        files.update(value["validated_dependency_files"])
    plan = {
        **config,
        "kind": "bounded_intervention_campaign",
        "candidates": programs,
        "seed_history": history,
        "files": files,
        "policy": "bounded_priority_v1",
        "skill_scope": "offline_training_service",
        "automatic_promotion": False,
        "navtrain_student_scoring": False,
        "policy_gain_credited": False,
    }
    plan["campaign_id"] = identity(plan)
    out.mkdir(parents=True)
    write_json(out / "campaign.json", plan)
    return plan


def advance(output, *, execute=False):
    """Advance at most one native pair. A replay never executes a completed pair."""
    out = augustus_round._exp(output)
    with (out / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read_mapping(out / "campaign.json")
        if plan["campaign_id"] != identity({k: v for k, v in plan.items() if k != "campaign_id"}):
            raise ValueError("Campaign configuration changed")
        augustus_round._verify(plan["files"])
        receipts = []
        seen = set()
        for index, p in enumerate(sorted(out.glob("trial-*/result.json"))):
            if p.parent.name != "trial-%03d" % index:
                raise ValueError("Trial receipts are not sequential")
            receipt = read_mapping(p)
            if receipt["campaign_id"] != plan["campaign_id"]:
                raise ValueError("Foreign trial receipt")
            augustus_round._verify(receipt["evidence"])
            assignment = read_mapping(p.parent / "assignment.json")
            candidate = assignment["candidate"]
            if (
                assignment["campaign_id"] != plan["campaign_id"]
                or candidate not in plan["candidates"]
                or candidate["id"] in seen
                or assignment["decision"].get("intervention_id") != candidate["id"]
            ):
                raise ValueError("Trial assignment differs from sealed campaign")
            seen.add(candidate["id"])
            review_path = Path(candidate["program"]) / "review.json"
            for ref, sha in augustus_round._files(
                [review_path, p.parent / "assignment.json"]
            ).items():
                if receipt["evidence"].get(ref) != sha:
                    raise ValueError("Receipt omits its actual review/assignment")
            review = read_mapping(review_path)
            expected_evidence = {
                **augustus_round._files([review_path, p.parent / "assignment.json"]),
                **review.get("execution_evidence", {}),
            }
            if receipt["evidence"] != expected_evidence:
                raise ValueError("Receipt execution evidence closure differs from review")
            augustus_round._verify(expected_evidence)
            observation = {
                "intervention_id": candidate["id"],
                "node": review["node"],
                "status": review["status"],
                "observed_gain": review.get("observed_gain"),
            }
            if (
                receipt["observation"] != observation
                or receipt["optimizer_update_reservation"]
                != candidate["optimizer_update_reservation"]
            ):
                raise ValueError("Receipt decision or budget differs from sealed evidence")
            receipts.append(receipt)
        history = plan["seed_history"] + [x["observation"] for x in receipts]
        used = sum(x["optimizer_update_reservation"] for x in receipts)
        decision = select_intervention(
            plan["candidates"], history, plan["max_trials"] - len(receipts)
        )
        if decision["action"] == "execute":
            candidate = next(
                c for c in plan["candidates"] if c["id"] == decision["intervention_id"]
            )
            if (
                used + candidate["optimizer_update_reservation"]
                > plan["max_optimizer_updates_including_replays"]
            ):
                decision = {"action": "stop", "reason": "optimizer_budget_exhausted"}
        if decision["action"] != "execute":
            summary = {
                "status": decision["action"],
                "decision": decision,
                "completed_trials": len(receipts),
                "reserved_optimizer_updates": used,
                "promoted": False,
                "policy_gain_credited": False,
                "model_parent_unchanged": True,
            }
            write_json(out / "summary.json", summary)
            return summary
        trial = out / ("trial-%03d" % len(receipts))
        assignment = {
            "campaign_id": plan["campaign_id"],
            "decision": decision,
            "candidate": candidate,
        }
        if (trial / "assignment.json").exists():
            if read_mapping(trial / "assignment.json") != assignment:
                raise ValueError("Interrupted assignment differs from current decision")
        else:
            trial.mkdir(exist_ok=True)
            write_json(trial / "assignment.json", assignment)
        if not execute:
            return {"status": "prepared", "assignment": assignment, "training_started": False}
        result = augustus_round.run(candidate["program"], execute=True)
        review_path = Path(candidate["program"]) / "review.json"
        if not review_path.exists():
            return {"status": "running", "assignment": assignment, "program_status": result}
        # Infrastructure failures remain evidence but never look like model failure.
        if result["node"] == "learning_review":
            result = checked_review(review_path)
        else:
            augustus_round._verify(result.get("execution_evidence", {}))
        observation = {
            "intervention_id": candidate["id"],
            "node": result["node"],
            "status": result["status"],
            "observed_gain": result.get("observed_gain"),
        }
        receipt = {
            "campaign_id": plan["campaign_id"],
            "observation": observation,
            "optimizer_update_reservation": candidate["optimizer_update_reservation"],
            "reserved_not_measured_cost": True,
            "evidence": augustus_round._files([review_path, trial / "assignment.json"]),
            "promoted": False,
            "policy_gain_credited": False,
        }
        receipt["evidence"].update(result.get("execution_evidence", {}))
        write_json(trial / "result.json", receipt)
        summary = {
            "status": "trial_completed",
            "completed_trials": len(receipts) + 1,
            "last_observation": observation,
            "reserved_optimizer_updates": used + candidate["optimizer_update_reservation"],
            "promoted": False,
            "policy_gain_credited": False,
            "model_parent_unchanged": True,
        }
        write_json(out / "summary.json", summary)
        return summary
