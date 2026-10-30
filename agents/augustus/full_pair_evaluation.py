"""Candidate evaluation on the identical completed full-parent chunk protocol."""

from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest, path_at, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from driversi_agents.augustus.evaluation import evaluate
from driversi_agents.augustus.full_evaluation import aggregate_rows, check_plan, completed_chunk


def run(plan, previous):
    source = Path(plan["data"]["full_evaluation_plan"])
    frozen = check_plan(source)
    parent = path_at(project_root(), plan["model_config"]["weights"]["parent"])
    if parent.resolve() != Path(frozen["config"]["checkpoint"]).resolve():
        raise ValueError("Training parent differs from full baseline checkpoint")
    parent_weight = str(parent / "model.safetensors")
    if frozen["inputs"].get(parent_weight) != digest(parent_weight):
        raise ValueError("Training parent bytes differ from full baseline")
    baseline = read_mapping(Path(frozen["output"]) / "report.json")
    if not baseline["complete"] or baseline["plan_sha256"] != digest(source):
        raise ValueError("Full completed parent evaluation required")
    out = Path(plan["output"])
    all_rows = {"parent": [], "candidate": []}
    arrays = {"parent": [], "candidate": []}
    tokens = {"parent": [], "candidate": []}
    spent = 0
    req = plan["resolved"]["request"]
    if req["budget"]["max_scoring_agent_steps"] < 80 * len(frozen["records"]):
        raise ValueError("Insufficient full candidate scoring budget")
    from driversi_agents.augustus.candidate_dispatch import precompute

    precompute(plan, previous, frozen)
    for i in range(len(frozen["chunks"])):
        chunk = Path(frozen["output"]) / "chunks" / f"{i:04d}"
        parent = completed_chunk(frozen, i)
        if parent is None:
            raise ValueError("Missing parent chunk")
        candidate = evaluate(
            chunk / "data/receipt.json",
            previous["checkpoint"],
            read_mapping(chunk / "context/receipt.json")["metric_cache"],
            out / "candidate-chunks" / f"{i:04d}",
            seed=frozen["config"]["seed"] + i,
            batch_size=frozen["config"]["batch_size"],
            reuse_model=True,
        )
        parent_path = Path(read_mapping(chunk / "completed.json")["report"]).parent
        for name, result, folder in (
            ("parent", parent, parent_path),
            ("candidate", candidate, out / "candidate-chunks" / f"{i:04d}"),
        ):
            all_rows[name].extend(result["rows"])
            with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
                arrays[name].append(archive["trajectories"])
                tokens[name].extend(archive["tokens"].tolist())
        spent += read_mapping(out / "candidate-chunks" / f"{i:04d}" / "scoring-budget.json")[
            "reserved_agent_steps"
        ]
        write_json(
            out / "evaluation-progress.json",
            dict(completed=len(all_rows["candidate"]), expected=12146),
        )
    metrics = {}
    pairs = {}
    for name in ("parent", "candidate"):
        summary = aggregate_rows(all_rows[name], frozen["records"])
        if not summary["complete"]:
            raise ValueError("Partial test is not a result")
        metrics[name] = summary["metrics"]
        pairs[name] = {x["token"]: x for x in all_rows[name]}
        np.savez_compressed(
            out / (name + "-pdm-inputs.npz"),
            trajectories=np.concatenate(arrays[name]),
            tokens=np.asarray(tokens[name], dtype=str),
        )
        write_json(out / (name + "-pdm-samples.json"), all_rows[name])
    if spent > req["budget"]["max_scoring_agent_steps"]:
        raise ValueError("Candidate scoring budget exceeded")
    return dict(
        status="completed",
        reason=None,
        metrics=metrics,
        directions={k: 1 for k in ("PDMS", "NC", "DAC", "DDC", "TTC", "C")},
        primary_metric="PDMS",
        safety_higher=["NC", "DAC", "DDC", "TTC"],
        metric_scope="native_navsim_pdm_on_full_official_navtest",
        evaluated_sample_ids=tokens["parent"],
        paired_samples=[
            dict(
                sample_id=t,
                source_log=pairs["parent"][t]["source_log"],
                parent=pairs["parent"][t]["metrics"],
                candidate=pairs["candidate"][t]["metrics"],
            )
            for t in tokens["parent"]
        ],
        evaluated_checkpoint=previous["checkpoint"],
        parent_evaluation_reused=True,
        parent_evaluation_sha256=digest(Path(frozen["output"]) / "report.json"),
        consumed=dict(environment_steps=0, scoring_agent_steps=spent),
    )
