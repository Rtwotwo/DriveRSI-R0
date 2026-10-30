"""Bounded autonomous curriculum -> training -> independent research promotion."""

import socket
import time
from pathlib import Path

from driversi.configuration import read_mapping
from driversi.runtime.budget import BudgetExceeded, write_json
from driversi.rsi.experiments.campaign_spec import resolve_campaign
from driversi.rsi.experiments.campaign_store import CampaignStore
from driversi.rsi.contracts import BUDGET_FIELDS
from driversi.rsi.execution.executor import children_cpu, execute, recover
from driversi.rsi.decisions.improver import propose, validate_proposal
from driversi.rsi.execution.jobs import process_identity
from driversi.rsi.execution.worker import verify_sources
from driversi.training.sampling.environment import validate_development_paths


def register_campaign(registry, spec_path, output_root):
    spec_path = Path(spec_path)
    return CampaignStore(registry).register(
        resolve_campaign(read_mapping(spec_path), spec_path.parent), output_root
    )


def run_campaign(registry, campaign_id):
    store = CampaignStore(registry)
    state = store.acquire(campaign_id)
    spec = state["spec"]
    directory = Path(state["directory"])
    owned = False
    started, own_cpu, child_cpu = time.monotonic(), time.process_time(), children_cpu()

    def refresh():
        costs = dict.fromkeys(BUDGET_FIELDS, 0)
        for jid in state["job_ids"]:
            try:
                job = store.jobs.get(jid)
            except ValueError:
                continue
            for key in costs:
                costs[key] += job["cost"][key]
        costs["cpu_seconds"] = max(
            costs["cpu_seconds"], time.process_time() - own_cpu + children_cpu() - child_cpu
        )
        state["cost"] = costs
        state["elapsed_seconds"] = time.monotonic() - started

    def check():
        if store.get(campaign_id)["cancel_requested"]:
            raise InterruptedError("operator_cancelled")
        if state["elapsed_seconds"] >= spec["wall_seconds"] or any(
            state["cost"][k] > spec["budget"][k] for k in BUDGET_FIELDS
        ):
            raise BudgetExceeded("campaign resource/deadline limit")

    def phase(kind, train_paths, eval_paths, weights, reference=None, recipe=None):
        refresh()
        check()
        verify_sources(spec)
        i = state["current"]["index"]
        jid = f"{campaign_id}.r{i}.{kind}"
        plan = {
            "schema_version": 1,
            "id": jid,
            "target": "model",
            "hypothesis": f"Bounded campaign {kind}; research only",
            "arms": {
                "baseline": {
                    "version": "reference",
                    "initial_model": state["incumbent"]["version"],
                    "information_set": campaign_id,
                    "tools": "campaign-v1",
                },
                "candidate": {
                    "version": "proposal",
                    "initial_model": state["incumbent"]["version"]
                    if kind != "gate"
                    else state["current"]["candidate"]["version"],
                    "information_set": campaign_id,
                    "tools": "campaign-v1",
                },
            },
            "protocol": {
                "id": spec["gate"]["id"],
                "evaluator_version": "driversi.teacher.v1",
                "role": "promotion" if kind == "gate" else "development",
                "task_set": jid,
                "metric": spec["gate"]["metric"],
                "direction": spec["gate"]["direction"],
            },
            "data": {
                "development_groups": sorted(validate_development_paths(train_paths)),
                "evaluation_groups": sorted(validate_development_paths(eval_paths)),
            },
            "budget_per_arm": spec["stage_budget"][kind],
            "scene_ids": [] if kind == "gate" else [Path(p).stem for p in train_paths + eval_paths],
        }
        store.registry.register(plan)
        inputs = directory / "inputs"
        suite = inputs / f"{jid}.suite.json"
        write_json(
            suite,
            {
                "official_test": False,
                "train": [{"path": p} for p in train_paths],
                "evaluation": [{"path": p} for p in eval_paths],
            },
        )
        job = {
            "schema_version": 1,
            "id": jid,
            "plan_id": jid,
            "arm": "candidate",
            "tool": {
                "mining": "structured_probe",
                "training": "structured_ppo",
                "gate": "structured_gate",
            }[kind],
            "suite": str(suite),
            "agent_config": str(inputs / "agent.json"),
            "seed": spec["seed"] + i,
            "device": spec["device"] if kind == "training" else "cpu",
            "gpu_id": spec["gpu_id"],
            "wall_seconds": max(0.001, spec["wall_seconds"] - state["elapsed_seconds"]),
            "initial_weights": weights,
            "target_transitions": 0,
            "max_train_agent_steps": 0,
            **spec["training"],
        }
        if recipe:
            job.update(recipe)
        if reference:
            job.update(reference_weights=reference["weights"], reference_model=reference["version"])
        path = inputs / f"{jid}.job.json"
        write_json(path, job)
        state["job_ids"].append(jid)
        state["current"][kind + "_job"] = jid
        state["phase"] = kind
        store.update(state)
        result = execute(
            registry,
            path,
            directory / "runs",
            cancel_requested=lambda: store.get(campaign_id)["cancel_requested"],
        )
        refresh()
        store.update(state)
        if result["status"] != "completed":
            if result["status"] == "cancelled":
                raise InterruptedError(result["failure_reason"])
            if result["status"] == "budget_exhausted":
                raise BudgetExceeded(result["failure_reason"])
            raise RuntimeError(result["failure_reason"])
        check()
        return result

    try:
        directory.mkdir(parents=True, exist_ok=False)
        owned = True
        (directory / "inputs").mkdir()
        write_json(directory / "inputs/agent.json", spec["agent"])
        write_json(directory / "campaign.json", spec)
        for i in range(spec["rounds"]):
            refresh()
            check()
            reserved = {k: sum(b[k] for b in spec["stage_budget"].values()) for k in BUDGET_FIELDS}
            if any(state["cost"][k] + reserved[k] > spec["budget"][k] for k in BUDGET_FIELDS):
                raise BudgetExceeded(
                    "insufficient budget to reserve complete mining/training/gate round"
                )
            state["current"] = {"index": i, "parent": dict(state["incumbent"])}
            mined = phase(
                "mining",
                spec["train_paths"],
                spec["development_paths"],
                state["incumbent"]["weights"],
            )
            parent_weights = str(Path(mined["directory"]) / "model-weights.pt")
            if state["incumbent"]["weights"] is None:
                state["incumbent"]["weights"] = parent_weights
                state["current"]["parent"] = dict(state["incumbent"])
            # Proposer input contains development episodes only, never gate output or config.
            proposal = validate_proposal(
                propose(mined["evaluation"], spec["improver"], spec["recipes"], spec["seed"] + i),
                spec["train_paths"],
                spec["recipes"],
            )
            state["current"]["proposal"] = proposal
            state["phase"] = "proposal"
            store.update(state)
            write_json(directory / f"proposal-{i}.json", proposal)
            trained = phase(
                "training",
                proposal["selected_paths"],
                spec["development_paths"],
                state["incumbent"]["weights"],
                recipe=spec["recipes"][proposal["recipe"]],
            )
            state["current"]["candidate"] = {
                "version": trained["job"]["candidate_version"],
                "weights": str(Path(trained["directory"]) / "model-weights.pt"),
            }
            phase(
                "gate",
                spec["train_paths"] + spec["development_paths"],
                spec["promotion_slots"][i],
                state["current"]["candidate"]["weights"],
                reference=state["incumbent"],
            )
            refresh()
            check()
            store.update(state)
            state = store.close_round(campaign_id)
            if state["status"] != "running":
                break
        if state["status"] == "running":
            refresh()
            check()
            state.update(status="completed", phase="completed")
    except InterruptedError as exc:
        state.update(status="cancelled", phase="cancelled", failure_reason=str(exc))
    except BudgetExceeded as exc:
        state.update(status="budget_exhausted", phase="budget_exhausted", failure_reason=str(exc))
    except Exception as exc:
        state.update(status="failed", phase="failed", failure_reason=f"{type(exc).__name__}: {exc}")
    except BaseException:
        state.update(status="cancelled", phase="cancelled", failure_reason="controller_interrupted")
    finally:
        refresh()
        # An uncleared child reservation cannot be hidden by a terminal campaign.
        active = []
        for jid in state["job_ids"]:
            try:
                if store.jobs.get(jid)["status"] in ("queued", "running"):
                    active.append(jid)
            except ValueError:
                pass
        if active:
            state.update(
                status="running",
                phase="recovery_required",
                failure_reason="owned job still has reservation",
            )
        state["finished_at"] = time.time()
        if store.get(campaign_id)["status"] == "running":
            store.update(state)
        if owned:
            write_json(directory / "result.json", state)
    return state


def recover_campaign(registry, campaign_id):
    store = CampaignStore(registry)
    state = store.get(campaign_id)
    owner = state["owner"]
    if state["status"] != "running" or not owner or owner["host"] != socket.gethostname():
        raise ValueError("Only an interrupted campaign on its original host can recover")
    current = process_identity(owner["pid"])
    if current is not None and (owner["start"] is None or current == owner["start"]):
        raise ValueError("Controller still alive; cancel instead")
    costs = dict.fromkeys(BUDGET_FIELDS, 0)
    for jid in state["job_ids"]:
        try:
            job = store.jobs.get(jid)
        except ValueError:
            continue
        if job["status"] in ("queued", "running"):
            job = recover(store.jobs, jid)
        for k in costs:
            costs[k] += job["cost"][k]
    costs["cpu_seconds"] = max(costs["cpu_seconds"], state["spec"]["budget"]["cpu_seconds"])
    state.update(
        status="failed",
        phase="failed",
        cost=costs,
        failure_reason="controller_lost_unknown_control_cpu_charged",
        finished_at=time.time(),
    )
    store.update(state)
    return state
