"""Allowlisted single-process tools. Evaluation is launched as a separate invocation."""
import argparse
import ctypes
import json
import math
import os
import resource
import signal
import sys
import traceback
from pathlib import Path

from driversi.runtime.budget import StepBudget, write_json


def verify_sources(job):
    for item in job["sources"]:
        stat = Path(item["path"]).stat()
        if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError("Scene changed after job admission")


def train_or_sample(job, directory):
    import numpy as np
    import torch

    from driversi.agents.interfaces.config import build_agent
    from driversi.data.scenes.scene_store import source_inventory
    from driversi.training.sampling.collector import Collector
    from driversi.training.ppo import update

    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    torch.use_deterministic_algorithms(True)
    model = build_agent(job["agent"]).to(job["device"])
    if job.get("initial_weights"):
        payload = torch.load(job["initial_weights"], map_location="cpu", weights_only=True)
        if payload.get("agent") != job["agent"] or payload.get("schema_version") != 1:
            raise ValueError("Portable initialization agent contract mismatch")
        model.load_state_dict(payload["model"], strict=True)
    model.export(directory / "initial.npz", job["initial_model"])
    guard = StepBudget(directory / "progress.json", job["max_train_agent_steps"], "training")
    if job["tool"] in ("structured_probe", "structured_gate"):
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError("Non-finite model")
        model.export(directory / "candidate.npz", job["candidate_version"])
        torch.save({"schema_version": 1, "agent": job["agent"], "model": model.state_dict(),
                    "policy_version": job["candidate_version"]}, directory / "model-weights.pt")
        if job["tool"] == "structured_gate":
            reference = build_agent(job["agent"])
            payload = torch.load(job["reference_weights"], map_location="cpu", weights_only=True)
            if payload.get("schema_version") != 1 or payload.get("agent") != job["agent"]:
                raise ValueError("Reference agent contract mismatch")
            reference.load_state_dict(payload["model"], strict=True)
            if not all(torch.isfinite(p).all().item() for p in reference.parameters()):
                raise ValueError("Non-finite reference model")
            reference.export(directory / "reference.npz", job["reference_model"])
        write_json(directory / "training-result.json", {"status": "candidate_ready", "tool": job["tool"],
                   "valid_transitions": 0, "updates": 0, "batches": 0, "approved_teacher": False})
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=job["learning_rate"], eps=1e-5)
    total = updates = batches = 0
    collector = Collector(job["train_paths"], job["worlds"], job["seed"],
                          workers=0, max_rollout_mib=job["max_rollout_mib"])
    try:
        write_json(directory / "scene-inventory.json",
                   source_inventory(collector.paths, collector.progress_weight))
        while total < job["target_transitions"]:
            policy_version = f"{job['candidate_version']}.update{updates}"
            if job["recording"] == "full":
                model.export(directory / f"policy-{batches:05d}.npz", policy_version)
            rollout = collector.collect(
                model, job["rollout_steps"], job["device"], policy_version,
                before_step=guard.before_step)
            count = int(rollout["valid"].sum())
            if not count:
                raise ValueError("No valid training transitions")
            stats = {}
            if job["tool"] == "structured_ppo":
                stats = update(model, optimizer, rollout, epochs=job["epochs"],
                               minibatch=job["minibatch_size"], device=job["device"])
                updates += 1
            if job["recording"] == "full":
                np.savez_compressed(
                    directory / f"rollout-{batches:05d}.npz",
                    metadata=json.dumps({"schema_version": 1, "kind": "rsi-structured-rollout",
                                         "policy_version": policy_version, "agent": job["agent"],
                                         "source_manifest": "scene-inventory.json",
                                         "approved_teacher": False}), **rollout)
            total += count
            batches += 1
            guard.progress(valid_transitions=total, updates=updates, batches=batches,
                           last_update=stats)
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError("Candidate contains non-finite parameters")
        model.export(directory / "candidate.npz", job["candidate_version"])
        torch.save({"schema_version": 1, "agent": job["agent"], "model": model.state_dict(),
                    "policy_version": job["candidate_version"]}, directory / "model-weights.pt")
        result = {"status": "candidate_ready", "tool": job["tool"],
                  "valid_transitions": total, "updates": updates, "batches": batches,
                  "candidate_model": job["candidate_version"], "approved_teacher": False,
                  "sampling_keeps_initial_weights": job["tool"] == "structured_sample"}
        write_json(directory / "training-result.json", result)
    finally:
        collector.close()


def evaluate_candidate(job, directory):
    from driversi.agents.interfaces.config import load_inference
    from driversi.evaluation.search.teacher import evaluate

    guard = StepBudget(directory / "progress.json",
                       job["reserved"]["sim_agent_steps"], "evaluating")
    policy = load_inference(directory / "candidate.npz", job["agent"])
    probe = job["tool"] == "structured_probe"
    result = evaluate(policy, job["train_paths"] if probe else job["eval_paths"],
                      training_paths=None if probe else job["train_paths"],
                      before_step=guard.before_step,
                      on_episode=lambda n: guard.progress(evaluated_scenes=n))
    if job["tool"] == "structured_gate":
        reference = load_inference(directory / "reference.npz", job["agent"])
        result["reference"] = evaluate(reference, job["eval_paths"], training_paths=job["train_paths"],
                                       before_step=guard.before_step)
        result["reference_model"] = job["reference_model"]
    result["purpose"] = ("development_mining" if probe else
                         "promotion_pair" if job["tool"] == "structured_gate" else "development_evaluation")
    result.update(evaluator_version=job["evaluator_version"],
                  task_set=job["task_set"], candidate_model=job["candidate_version"],
                  official_score=False, approved_teacher=False,
                  independent_process=True)
    write_json(directory / "evaluation-result.json", result)
    guard.progress(phase="evaluated")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("stage", choices=("training", "evaluating"))
    parser.add_argument("--cpu-limit", type=float, required=True)
    parser.add_argument("--supervisor-pid", type=int)
    args = parser.parse_args(argv)
    if args.supervisor_pid is not None:
        if os.getppid() != args.supervisor_pid:
            return 3
        if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0:
            raise RuntimeError("Cannot bind tool lifetime to supervisor")
        if os.getppid() != args.supervisor_pid:
            return 3
    job = json.loads(args.job.read_text())
    directory = args.job.parent
    # Kernel backup to the supervisor's finer-grained CPU accounting.
    seconds = max(1, math.ceil(args.cpu_limit))
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
    try:
        verify_sources(job)
        if args.stage == "training":
            train_or_sample(job, directory)
        else:
            evaluate_candidate(job, directory)
    except Exception as exc:
        write_json(directory / "worker-error.json",
                   {"type": type(exc).__name__, "message": str(exc), "stage": args.stage})
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
