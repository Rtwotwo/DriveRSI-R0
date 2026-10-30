"""Unified job lifecycle; native algorithms and official evaluators stay independent."""

import argparse
import json
import time
from pathlib import Path

import yaml

from driversi.agents.backends import native
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from driversi.training.contracts.job import resolve_request
from driversi.training.sampling.environment import source_log_group


def _sources(config, registry):
    """Use the existing backend manifest validators, never invent data splits."""
    root = native.project_root()
    protected = set()
    if registry is not None:
        from driversi.rsi.execution.registry import ExperimentRegistry

        protected = ExperimentRegistry(registry).protected_groups()
    official = (
        root
        / "src/vendor/driving/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )
    if official.is_file():
        protected |= set(read_mapping(official)["log_names"])
    groups = []
    if config["model"] == "augustus":
        path = native.path_at(root, config["paths"]["dataset_manifest"])
        if path.is_file():
            from driversi_agents.augustus.data import manifest

            data = manifest(path)
            groups = [r["log_name"] for b in data["splits"].values() for r in b["records"]]
            groups += data["parent_training_logs"]
    if {source_log_group(x) for x in groups} & {source_log_group(x) for x in protected}:
        raise ValueError("Protected registry source in native training")
    return sorted(set(groups))


def prepare(request_path, output, *, registry=None):
    resolved = resolve_request(request_path)
    if resolved["request"]["schema_version"] in (2, 3, 4):
        from driversi.training.orchestration.lifecycle import prepare as prepare_lifecycle
        return prepare_lifecycle(resolved, output)
    output = Path(output).resolve()
    if not output.is_relative_to(native.project_root() / "exp"):
        raise ValueError("New training artifacts belong under project exp")
    if output.exists():
        raise ValueError("Training output must be new; use run on its saved plan")
    config = read_mapping(resolved["model_config"])
    budget = resolved["request"]["budget"]
    resolved["source_groups"] = _sources(config, registry) if resolved["backend"] else []
    output.mkdir(parents=True)
    plan = {
        "schema_version": 1,
        "resolved": resolved,
        "output": str(output),
        "registry": str(Path(registry).resolve()) if registry else None,
        "native_plan": None,
        "ready_to_attempt": False,
        "missing": [] if resolved["backend"] else [resolved["reason"]],
    }
    if resolved["backend"]:
        config["settings"]["max_optimizer_steps"] = budget["max_optimizer_steps"]
        if resolved["model"] == "augustus":
            config["settings"]["max_sim_agent_steps"] = budget["max_scoring_agent_steps"]
        elif budget["max_scoring_agent_steps"]:
            raise ValueError("World-model backend does not spend driving-score queries")
        path = output / "native-model.yaml"
        write_json(output / "request.json", resolved["request"])
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        plan["native_plan"] = native.prepare(path, "train", output / "native")
        if plan["native_plan"]["protocol"] != resolved["backend"]["native_protocol"]:
            raise ValueError("Native learning objective differs from requested backend")
        plan["missing"] = plan["native_plan"]["missing"]
        plan["ready_to_attempt"] = plan["native_plan"]["ready_to_attempt"]
    write_json(output / "training-plan.json", plan)
    return plan


def _receipt(plan):
    r = plan["resolved"]
    return {
        "schema_version": 1,
        "task_id": r["request"]["id"],
        "model": r["model"],
        "family": r["family"],
        "mode": r["request"]["mode"],
        "backend": r["backend"],
        "status": "not_started",
        "reason": None,
        "checkpoint": None,
        "consumed": {
            "wall_seconds": 0.0,
            "gpu_seconds": 0.0,
            "environment_steps": 0,
            "scoring_agent_steps": 0,
            "native_global_steps": None,
            "successful_optimizer_steps": None,
        },
        "initialization": r["initialization"],
        "exact_resume_supported": False,
        "evaluation_status": "not_evaluated",
        "promoted": False,
        "policy_activated": False,
        "performance_parity": "not_measured",
        "source_groups": r["source_groups"],
    }


def _save_attempt(output, plan, report):
    from driversi.rsi.decisions.training_diagnosis import diagnose_training_attempt

    report["diagnosis"] = diagnose_training_attempt(plan, report)
    write_json(Path(output) / "training-result.json", report)


def run(plan_path, *, own_process_group=True):
    start = time.monotonic()
    plan = read_mapping(plan_path)
    if plan["schema_version"] == 2:
        from driversi.training.orchestration.lifecycle import execute
        return execute(plan, own_process_group=own_process_group)
    output = Path(plan["output"])
    if (output / "training-result.json").exists() or (output / "native").exists():
        raise ValueError("Training attempt already exists; do not repeat side effects")
    r = plan["resolved"]
    if resolve_request(r["request_path"]) != {k: v for k, v in r.items() if k != "source_groups"}:
        raise ValueError("Training declaration changed after preparation")
    config = read_mapping(r["model_config"])
    if r["backend"] and _sources(config, plan["registry"]) != r["source_groups"]:
        raise ValueError("Training source declaration changed")
    report = _receipt(plan)
    if not plan["ready_to_attempt"]:
        report.update(
            status="rejected", reason="native_prerequisites_missing", missing=plan["missing"]
        )
        _save_attempt(output, plan, report)
        return report
    write_json(output / "training-result.json", {**report, "status": "running"})
    budget = r["request"]["budget"]
    report["wall_budget_scope"] = "verification_execution_artifact_receipt_excluding_preparation"
    report["counter_semantics"] = (
        "native_global_steps_may_include_skipped_batches_not_gradient_evidence"
    )
    try:
        outcome = native.execute(
            plan["native_plan"],
            timeout_seconds=budget["max_wall_seconds"],
            own_process_group=own_process_group,
            deadline_monotonic=start + budget["max_wall_seconds"],
        )
        report["status"] = outcome["status"]
        report["reason"] = outcome.get("reason")
        if report["status"] == "failed" and not report["reason"]:
            report["reason"] = "native_process_failed: returncode=" + str(outcome.get("returncode"))
        report["native_execution"] = {
            k: outcome.get(k) for k in ("returncode", "native_log_sha256", "plan_sha256")
        }
        report["native_execution"]["log"] = str(output / "native/native.log")
        report["consumed"]["gpu_seconds"] = outcome["reserved_gpu_seconds"]
        native_dir = output / "native"
        if r["model"] == "augustus":
            from driversi.rsi.plugins.augustus import consumed

            report["consumed"]["scoring_agent_steps"] = consumed(native_dir)
        if outcome["status"] == "completed":
            info = read_mapping(
                native_dir
                / ("candidate.json" if r["model"] == "augustus" else "training-receipt.json")
            )
            steps = info["updates"] if r["model"] == "augustus" else info["native_global_steps"]
            if type(steps) is not int or not 0 < steps <= budget["max_optimizer_steps"]:
                raise ValueError("Native global counter violates the declared budget")
            checkpoint = Path(info["checkpoint"])
            if not checkpoint.resolve().is_relative_to(
                Path(plan["native_plan"]["checkpoint_output"]).resolve()
            ):
                raise ValueError("Checkpoint is outside this native training attempt")
            files = {
                p: native.digest(p) for p in native.asset_files(checkpoint) if Path(p).is_file()
            }
            if not files:
                raise ValueError("Native checkpoint artifact missing")
            report["consumed"]["native_global_steps"] = steps
            observed = info.get("update_evidence", {})
            if observed.get("observer") in (
                "native_optimizer_post_step_hook",
                "deepspeed_engine_update_counters",
            ):
                updates = observed.get("successful_optimizer_steps")
                if type(updates) is not int or not 0 <= updates <= steps:
                    raise ValueError("Observed optimizer updates violate native step evidence")
                report["consumed"]["successful_optimizer_steps"] = updates
            report["checkpoint"] = {
                "path": str(checkpoint),
                "files": files,
                "format": "native_bundle",
                "native_receipt": info,
                "source_config": r["model_config"],
                "source_config_sha256": r["model_config_sha256"],
            }
            report["evaluation_status"] = "independent_evaluation_required"
        if report["consumed"]["scoring_agent_steps"] > budget["max_scoring_agent_steps"]:
            raise ValueError("Native scoring exceeded its reserved budget")
    except Exception as exc:
        report.update(status="failed", reason=type(exc).__name__ + ": " + str(exc))
        report["checkpoint"] = None
    finally:
        report["consumed"]["wall_seconds"] = time.monotonic() - start
        if (
            report["status"] == "completed"
            and report["consumed"]["wall_seconds"] > budget["max_wall_seconds"]
        ):
            report.update(status="budget_exhausted", reason="total_native_wall_budget_exhausted")
        _save_attempt(output, plan, report)
    return report


def run_online(request_path, output, *, session, learner, chunk_steps):
    """One frozen-policy collection/update round, with an explicitly native learner.

    No auto-reset, model swap, probability conversion or generic RL loss. Calls
    cooperate with wall checks; a process supervisor supplies a hard deadline.
    """
    from driversi.training.online_rl.online import Batch, update_from_batch

    r = resolve_request(request_path)
    if r["request"]["mode"] != "online_rl":
        raise ValueError("Live actor/learner execution requires online_rl mode")
    if type(chunk_steps) is not int or chunk_steps <= 0:
        raise ValueError("Positive collection chunk length required")
    if (
        getattr(session.actor, "model_id", None) != r["model"]
        or getattr(learner, "model_id", None) != r["model"]
    ):
        raise ValueError("Native actor and learner must identify the requested model")
    if (
        not isinstance(getattr(learner, "implementation_version", None), str)
        or not learner.implementation_version
    ):
        raise ValueError("Native learner implementation version required")
    output = Path(output).resolve()
    if not output.is_relative_to(native.project_root() / "exp"):
        raise ValueError("New training artifacts belong under project exp")
    output.mkdir(parents=True, exist_ok=False)
    r["source_groups"] = [session.source_log]
    report = _receipt({"resolved": r})
    report["backend"] = {
        "id": "explicit_native_online_learner",
        "implementation_version": learner.implementation_version,
    }
    report["initialization"] = "caller_supplied_live_actor_state"
    report["effective_parent"] = {
        "policy_version": session.version,
        "likelihood_contract": session.contract,
        "model_config_role": "declaration_only_not_applied_to_injected_actor",
    }
    report["consumed"]["gpu_seconds"] = None
    calls_before, confirmed_before = session.environment_step_calls, session.confirmed_transitions
    b = r["request"]["budget"]
    start, records, reasons = time.monotonic(), [], []
    candidate = None
    learner_invoked = False
    write_json(output / "request.json", r["request"])
    write_json(output / "training-result.json", {**report, "status": "running"})
    try:
        while len(records) < b["max_environment_steps"] and not session.ended:
            remaining = b["max_wall_seconds"] - (time.monotonic() - start)
            if remaining <= 0:
                reasons.append("collection_wall_budget_exhausted")
                break
            batch = session.collect(
                max_environment_steps=min(chunk_steps, b["max_environment_steps"] - len(records)),
                max_wall_seconds=remaining,
            )
            records.extend(batch.transitions)
            reasons.extend(batch.reasons)
            if reasons or not batch.transitions:
                break
        if time.monotonic() - start >= b["max_wall_seconds"]:
            reasons.append("collection_wall_budget_exhausted")
        combined = Batch(
            session.version,
            session.contract,
            session.source_log,
            tuple(records),
            session.observation,
            session.memory,
            tuple(sorted(set(reasons))),
            time.monotonic() - start,
        )
        write_json(output / "rollout-receipt.json", combined.receipt())

        class AccountedLearner:
            parent_version = learner.parent_version
            likelihood_contract = learner.likelihood_contract

            def update(self, batch, *, max_optimizer_steps):
                nonlocal learner_invoked
                learner_invoked = True
                return learner.update(batch, max_optimizer_steps=max_optimizer_steps)

        result = update_from_batch(
            AccountedLearner(), combined, max_optimizer_steps=b["max_optimizer_steps"]
        )
        report["status"] = result["status"]
        report["reason"] = ";".join(result["reasons"]) or None
        report["consumed"]["native_global_steps"] = result["optimizer_steps"]
        report["consumed"]["successful_optimizer_steps"] = result["optimizer_steps"]
        if result["status"] == "completed":
            candidate = Path(result["candidate"])
            if not candidate.resolve().is_relative_to(native.project_root() / "exp"):
                raise ValueError("Native online checkpoints belong under exp")
            files = {
                p: native.digest(p) for p in native.asset_files(candidate) if Path(p).is_file()
            }
            if not files:
                raise ValueError("Native learner checkpoint missing")
            report["checkpoint"] = {
                "path": str(candidate),
                "files": files,
                "format": "native_bundle",
                "parent_version": session.version,
                "likelihood_contract": session.contract,
            }
            report["evaluation_status"] = "independent_evaluation_required"
        if time.monotonic() - start > b["max_wall_seconds"]:
            report.update(status="budget_exhausted", reason="cooperative_wall_limit_exceeded")
    except Exception as exc:
        report.update(status="failed", reason=type(exc).__name__ + ": " + str(exc))
        report["checkpoint"] = None
    finally:
        elapsed = time.monotonic() - start
        calls = session.environment_step_calls - calls_before
        confirmed = session.confirmed_transitions - confirmed_before
        report["consumed"].update(
            wall_seconds=elapsed,
            environment_steps=calls,
            confirmed_transitions=confirmed,
            unknown_environment_outcomes=calls - confirmed,
            exported_transitions=len(records),
        )
        report["cost_semantics"] = (
            "environment_calls_including_unknown_outcomes_gpu_allocation_uninstrumented"
        )
        report["learner_invoked"] = learner_invoked
        if not learner_invoked:
            report["consumed"]["successful_optimizer_steps"] = 0
        if session.unresolved_step or (
            learner_invoked and report["consumed"]["successful_optimizer_steps"] is None
        ):
            report["status"] = "outcome_unknown"
        write_json(output / "training-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("request", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare")
    p.add_argument("request", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--registry", type=Path)
    sub.add_parser("run").add_argument("plan", type=Path)
    a = parser.parse_args()
    if a.command == "inspect":
        result = resolve_request(a.request)
        write_json(a.output, result)
    elif a.command == "prepare":
        result = prepare(a.request, a.output, registry=a.registry)
    else:
        result = run(a.plan)
    print(json.dumps(result, ensure_ascii=False))
    if a.command == "run" and result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
