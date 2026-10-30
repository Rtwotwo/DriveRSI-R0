"""Finite two-host experiment: full benchmark -> actual new pixels -> paired SFT.

These workers execute one declared experiment, not a recurring chat monitor.
Waiting for an upstream result cannot authorize relabelling or partial scoring.
"""

import argparse
import fcntl
import socket
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


def status(root, phase, state, **values):
    write_json(
        root / (phase + "-status.json"),
        dict(
            kind="reactive_experiment_worker",
            phase=phase,
            status=state,
            updated_at=time.time(),
            hostname=socket.gethostname(),
            **values,
        ),
    )


def wait_baseline(c, root, deadline):
    while time.time() < deadline:
        full = Path(read_mapping(c["baseline_config"])["output"]) / "baseline"
        if (full / "report.json").exists():
            report = read_mapping(full / "report.json")
            if report.get("complete") is True:
                return
            if report.get("status") in ("failed", "rejected"):
                raise ValueError("Baseline failed")
        value = read_mapping(full / "summary.json") if (full / "summary.json").exists() else {}
        if value.get("status") in ("failed", "rejected"):
            raise ValueError("Baseline failed")
        status(
            root,
            "production",
            "waiting_full_baseline",
            completed=value.get("completed", 0),
            expected=12146,
        )
        time.sleep(30)
    raise TimeoutError("Declared upstream wait budget exhausted")


def physical_prefetch_workers(config):
    """Bounded opt-in physics prefetch; completed jobs still settle in family order."""
    workers = config.get("physical_prefetch_workers", 1)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("physical_prefetch_workers must be an integer from1 to4")
    if workers != 1 and not config.get("trajectory_family"):
        raise ValueError("Physical prefetch requires a trajectory family")
    return workers


def production(config):
    from driversi.rsi.experiments.reactive_generation import generate
    from driversi.rsi.experiments.reactive_pair import compile_pair, train
    from driversi.rsi.experiments.reactive_postprocess import process
    from driversi.rsi.experiments.reactive_selection import select

    c = read_mapping(config)
    root = Path(c["output"])
    root.mkdir(parents=True,exist_ok=True)
    # Resuming an existing wave never acquires a fresh wall budget.
    deadline_path = root / 'production-deadline.json'
    from .continuation import frozen_json
    if not deadline_path.exists():
        start = min(time.time(), Path(config).stat().st_mtime)
        frozen_json(deadline_path, dict(started=start, deadline=start+c['max_wall_seconds'],
            max_wall_seconds=c['max_wall_seconds']))
    budget = read_mapping(deadline_path)
    if budget['max_wall_seconds'] != c['max_wall_seconds']:
        raise ValueError('Production time budget changed')
    deadline = budget['deadline']
    if time.time() >= deadline: raise TimeoutError('Original production wall budget exhausted')
    bounded = c.get("selection_mode") == "full_training_visual_bounded"
    wave = None
    if bounded:
        from driversi.rsi.decisions.augmentation_budget import runtime_gate

        wave = runtime_gate(c)
        write_json(root / "global-budget-gate.json", wave)
        if not wave["allowed"]:
            status(root, "production", "production_budget_blocked", reason=wave["reason"],
                   state_counts=wave["state_counts"], training_started=False,
                   generation_started=False)
            return
    wait_baseline(c, root, deadline)
    if c.get("selection_mode") in (
        "full_training_hybrid",
        "full_training_vectors",
        "full_training_clusters",
        "full_training_visual_bounded",
    ):
        from driversi.rsi.experiments.reactive_selection import select_full_training

        mining = Path(c["mining_root"])
        while not (mining / "summary.json").exists():
            if time.time() >= deadline:
                raise TimeoutError("Full-training retrieval wait budget exhausted")
            status(
                root, "production", "waiting_full_training_hybrid_mapping", training_members=103288
            )
            time.sleep(30)
        status(root, "production", "selecting_full_navtrain")
        excluded = list(c.get("exclude_training_tokens", []))
        if bounded:
            queue = read_mapping(mining / "training-production-queue.json")
            excluded.extend(r["token"] for r in queue["rows"] if r["token"] not in wave["selected_tokens"])
        selection = select_full_training(
            c["baseline_config"],
            mining,
            c["coverage"],
            root / "selection",
            max_candidates=c["max_candidates"],
            preferred_bundles=c.get("preferred_asset_bundles", []),
            exclude_tokens=excluded,
            generation_skill=c.get("generation_skill", "driverl_full_window"),
            **({"require_localization_free": True} if bounded else {}),
        )
    else:
        status(root, "production", "selecting_navval")
        selection = select(
            c["baseline_config"],
            c["coverage"],
            root / "selection",
            max_queries=c["max_queries"],
            max_candidates=c["max_candidates"],
            preferred_bundles=c.get("preferred_asset_bundles", []),
        )
    if c.get("trajectory_family"):
        from driversi.rsi.experiments.reactive_cohort import PendingGeneration, produce

        while True:
            try:
                ready = produce(
                    selection,
                    config,
                    root,
                    deadline,
                    status,
                    prefetch_workers=physical_prefetch_workers(c),
                )
                break
            except PendingGeneration:
                if time.time() >= deadline:
                    raise TimeoutError("Unsettled generation exceeded production budget")
                status(root, "production", "waiting_owned_generation_settlement")
                time.sleep(30)
        outcomes = read_mapping(root / "production-outcomes.json")["rows"]
    else:
        ready = []
        outcomes = []
        rendered = 0
        plan = read_mapping(selection["plan"])
        for assignment, asset in zip(plan["assignments"], selection["assets"], strict=True):
            if time.time() >= deadline:
                raise TimeoutError("Production wall budget exhausted")
            if len(ready) >= c["max_samples"] or rendered >= c["max_rendered_branches"]:
                break
            directory = root / "candidates" / assignment["id"]
            directory.mkdir(parents=True, exist_ok=True)
            status(
                root,
                "production",
                "generating_physical_branch",
                candidate=assignment["id"],
                completed_candidates=len(outcomes),
                qualified_samples=len(ready),
            )
            if (directory / "outcome.json").exists():
                outcome = read_mapping(directory / "outcome.json")
            else:
                try:
                    g = generate(
                        selection["plan"],
                        assignment["id"],
                        asset,
                        c["train_catalog"],
                        directory / "generation",
                        gpu=c.get("production_gpu", 6),
                    )
                    if g["status"] != "rendered":
                        outcome = dict(status=g["status"], trainable=False)
                    else:
                        outcome = process(
                            directory / "generation/report.json", config, directory / "admission"
                        )
                        outcome["rendered"] = True
                except Exception as exc:
                    # Preserve the attempted branch; never retry unknown physical/train side effects.
                    (directory / "failure.txt").write_text(traceback.format_exc())
                    outcome = dict(
                        status="execution_failed", error_type=type(exc).__name__, trainable=False
                    )
                write_json(directory / "outcome.json", outcome)
            render_receipt = directory / "generation/report.json"
            consumed_render = (directory / "generation/render-reservation.json").exists() or (
                render_receipt.exists() and read_mapping(render_receipt).get("status") == "rendered"
            )
            outcome["rendered"] = consumed_render
            write_json(directory / "outcome.json", outcome)
            rendered += int(consumed_render)
            if outcome.get("trainable") is True:
                ready.append(outcome["sample_report"])
            outcomes.append(dict(candidate=assignment["id"], **outcome))
            write_json(
                root / "production-outcomes.json",
                dict(rows=outcomes, qualified=len(ready), rendered=rendered),
            )
    if not ready:
        family_path = root / "trajectory-family-summary.json"
        exhausted = (
            family_path.exists()
            and read_mapping(family_path).get("status") == "review_budget_exhausted"
        )
        status(
            root,
            "production",
            "review_budget_exhausted" if exhausted else "no_qualified_generated_samples",
            qualified_samples=0,
            attempted=len(outcomes),
            training_started=False,
            fallback_to_relabel=False,
        )
        return
    if c.get("trajectory_family"):
        from driversi.rsi.experiments.reactive_cohort import family_training_gate

        declared = read_mapping(selection["plan"])
        gate = family_training_gate(
            read_mapping(root / "trajectory-family-summary.json"),
            [a["id"] for a in declared["assignments"]],
            c["trajectory_family"]["target_per_anchor"],
            **({'global_target':50} if c['trajectory_family'].get('cohort_mode')=='global_50_min1_v1' else {}),
        )
        write_json(root / "per-anchor-training-gate.json", gate)
        if not gate["passed"]:
            status(
                root,
                "production",
                "insufficient_per_anchor_samples",
                qualified_samples=len(ready),
                shortages=gate["shortages"],
                training_started=False,
            )
            return
    required = c.get("minimum_training_samples", 1)
    if len(ready) < required:
        status(
            root,
            "production",
            "insufficient_qualified_samples",
            qualified_samples=len(ready),
            required_samples=required,
            training_started=False,
            fallback_to_relabel=False,
        )
        return
    if bounded and c.get("enable_training_after_wave") is not True:
        status(root, "production", "generation_wave_completed", qualified_samples=len(ready),
               training_started=False, reason="separate_training_dispatch_required")
        return
    if time.time() >= deadline:
        raise TimeoutError("Production wall budget exhausted before training")
    status(root, "production", "preparing_native_pair", qualified_samples=len(ready))
    if c.get("student_model") == "chainflow_stage1":
        from driversi.rsi.experiments.chainflow_campaign import train_pair

        status(
            root,
            "production",
            "native_chainflow_pair_and_full_navtest",
            qualified_samples=len(ready),
        )
        ok = train_pair(config, ready, root / "chainflow-pair")
        status(
            root,
            "production",
            "completed" if ok else "training_failed",
            qualified_samples=len(ready),
            full_navtest_per_arm=12146,
        )
        return
    compile_pair(
        c["baseline_config"],
        ready,
        root / "pair",
        max_samples=c["max_samples"] if c.get("trajectory_family") else 4,
        optimizer_steps=c.get("pair_optimizer_steps", 4),
        worker_gpu=c.get("production_gpu", 6),
    )
    status(root, "production", "native_paired_training", qualified_samples=len(ready))
    ok = train(root / "pair/compiled.json")
    status(
        root,
        "production",
        "awaiting_full_candidate_evaluation" if ok else "training_failed",
        compiled=str(root / "pair/compiled.json"),
        qualified_samples=len(ready),
    )


def evaluation(config):
    from driversi.rsi.experiments.reactive_pair import evaluate_and_feedback

    c = read_mapping(config)
    root = Path(c["output"])
    deadline = time.time() + c["max_wall_seconds"]
    terminal = {
        "no_qualified_generated_samples",
        "insufficient_qualified_samples",
        "insufficient_per_anchor_samples",
        "training_failed",
        "failed",
        "review_budget_exhausted",
        "production_budget_blocked",
        "generation_wave_completed",
    }
    while time.time() < deadline:
        progress = (
            read_mapping(root / "production-status.json")
            if (root / "production-status.json").exists()
            else {}
        )
        if progress.get("status") in terminal:
            status(
                root,
                "evaluation",
                "upstream_rejected",
                reason=progress["status"],
                training_started=False,
            )
            return
        compiled = root / "pair/compiled.json"
        trained = root / "pair/training-status.json"
        if trained.exists() and read_mapping(trained).get("status") == "completed":
            status(root, "evaluation", "full_paired_navtest", expected_per_arm=12146)
            ok = evaluate_and_feedback(compiled)
            status(
                root,
                "evaluation",
                "completed" if ok else "failed",
                full_navtest=True,
                paired_feedback=str(root / "pair/paired-feedback.json"),
            )
            return
        status(
            root,
            "evaluation",
            "waiting_native_training",
            upstream=progress.get("status", "not_started"),
        )
        time.sleep(30)
    raise TimeoutError("Declared training wait budget exhausted")


def run(config, phase):
    config = Path(config).resolve(strict=True)
    c = read_mapping(config)
    root = Path(c["output"]).resolve()
    if not root.is_relative_to(project_root() / "exp"):
        raise ValueError("Experiment output must be under exp")
    if phase not in ("production", "evaluation"):
        raise ValueError("Invalid worker role")
    if socket.gethostname() != c[phase + "_host"]:
        raise ValueError("Experiment worker on wrong host")
    if c.get("trajectory_family"):
        from driversi.rsi.experiments.reactive_cohort import validate_budget

        validate_budget(c)
    elif not 1 <= c["max_samples"] <= 4 or not 1 <= c["max_rendered_branches"] <= 4:
        raise ValueError("Small actual-image pilot budget required")
    if not 60 <= c["max_wall_seconds"] <= 172800:
        raise ValueError("Finite experiment wall budget required")
    # Private credential values are neither copied nor added to experiment declarations.
    for key in (
        ()
        if c.get("student_model") == "chainflow_stage1"
        and c.get("visual_quality_profile")
        in (
            "experimental_numeric25_v1",
            "experimental_numeric24_ade03_v1",
            "experimental_numeric27_ade05_v1",
        "experimental_numeric27_strict_ade05_v1",
        "experimental_numeric25_spatial05_v1",
        "experimental_numeric25_adaptive_v1",
        )
        else ("critic_config", "credential_file")
    ):
        from driversi.data.catalog.bailian_caption import PRIVATE_ROOT
        if not Path(c[key]).resolve().is_relative_to(PRIVATE_ROOT.resolve()):
            raise ValueError("Credentials/config must stay in the private API directory")
    required = c.get("minimum_training_samples", 1)
    if type(required) is not int or not 1 <= required <= c["max_samples"]:
        raise ValueError("Invalid minimum qualified-sample requirement")
    physical_prefetch_workers(c)
    worker_gpu = c.get("production_gpu", 6)
    if type(worker_gpu) is not int or not 0 <= worker_gpu <= 7:
        raise ValueError("Invalid production GPU")
    root.mkdir(parents=True, exist_ok=True)
    frozen_json(
        root / "experiment.json",
        dict(
            config_sha256=digest(config),
            route="driverl_new_history_gs",
            inputs={p: digest(p) for p in (c["baseline_config"], c["coverage"])},
            original_anchor_relabel=False,
        ),
    )
    with ExitStack() as stack, (root / (phase + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if c.get("selection_mode") == "full_training_visual_bounded" and phase == "production":
            global_path = Path(c["global_budget"]).resolve(strict=True)
            if not global_path.is_relative_to(project_root() / "exp"):
                raise ValueError("Global budget must belong to experiment storage")
            lease = stack.enter_context((global_path.parent / "campaign.lock").open("a+"))
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            (production if phase == "production" else evaluation)(config)
        except BaseException as exc:
            (root / (phase + "-failure.txt")).write_text(traceback.format_exc())
            status(root, phase, "failed", error_type=type(exc).__name__)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("production", "evaluation"))
    a = parser.parse_args()
    run(a.config, a.phase)


if __name__ == "__main__":
    main()
