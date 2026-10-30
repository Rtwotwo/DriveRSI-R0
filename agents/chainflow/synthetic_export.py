"""Native-environment worker for an already admitted physical/rendered branch."""

import os
import pickle
import sys
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from driversi_agents.chainflow.synthetic import _sealed as check_files


def run(plan):
    from driversi.rsi.correction.trajectory_admission import validate_trajectory_evidence
    from driversi_agents.chainflow.synthetic import (
        export_native_frames,
        seal_admission,
        validate_visual_admission,
        write_counterfactual_caches,
    )

    check_files(plan["files"])
    branch = read_mapping(plan["branch"]["report"])
    validate_trajectory_evidence(branch["trajectory_evidence"])
    if not validate_visual_admission(plan["visual_admission"])["admitted"]:
        raise ValueError("Independent visual admission rejected this branch")
    root = project_root()
    sys.path[:0] = [
        str(root / "agents/chainflow/native"),
        str(root / "src/vendor/driving/nuplan-devkit"),
    ]
    os.environ["NUPLAN_MAPS_ROOT"] = plan["maps_root"]
    from navsim.common.numpy_pickle_compat import install_numpy_2_pickle_compat

    install_numpy_2_pickle_compat()
    # This pickle is an internal, hashed native runtime artifact, never an LLM
    # or remote user payload. The producer computed its reference before search.
    with Path(plan["context"]).open("rb") as handle:
        context = pickle.load(handle)
    if (
        context.get("kind") != "executed_counterfactual_context"
        or context.get("schema_version") != 1
    ):
        raise ValueError("Native counterfactual context is missing")
    required = {str(plan["branch"][k]): digest(plan["branch"][k]) for k in ("scene", "rollout")}
    if context["anchor_tick"] != branch["anchor_tick"]:
        raise ValueError("Cache context is not from this executed anchor")
    # Harness may copy an immutable producer artifact into a reference stage.
    # Accept only the explicit same-role, same-byte copy, and preserve its proof.
    source_files = context["branch_files"]
    if len(source_files) != len(required) or {Path(p).name for p in source_files} != {
        Path(p).name for p in required
    }:
        raise ValueError("Counterfactual artifact transport roles differ")
    transport = []
    for target, sha in required.items():
        candidates = [
            p for p, h in source_files.items() if Path(p).name == Path(target).name and h == sha
        ]
        if len(candidates) != 1 or digest(candidates[0]) != sha:
            raise ValueError("Counterfactual artifact transport contents differ")
        transport.append(dict(source=candidates[0], target=target, sha256=sha))
    out = Path(plan["output"])
    write_json(
        out / "context-transport.json",
        {"files": transport, "context_sha256": digest(plan["context"])},
    )
    from driversi.data.datasets.synthetic_storage import history_only_images, image_storage_layout
    image_contract = plan.get("image_contract")
    history_only = history_only_images(image_contract)
    if plan.get("canonical_dataset"):
        from driversi.data.datasets import synthetic_storage
        from driversi.data.datasets.synthetic_publish import publish_package
        from driversi_agents.chainflow import synthetic

        identity = dict(
            source_files={
                p: digest(p)
                for p in (plan["branch"]["scene"], plan["branch"]["rollout"], plan["render_report"])
            },
            future_points=10,
            exporter_sha256=digest(synthetic.__file__),
            storage_sha256=digest(synthetic_storage.__file__),
            layout=image_storage_layout(image_contract),
        )
        if history_only:
            identity["image_contract"] = image_contract
        exported = publish_package(
            identity,
            lambda staging: export_native_frames(
                plan["branch"]["scene"],
                plan["branch"]["rollout"],
                plan["render_report"],
                staging,
                anchor_tick=branch["anchor_tick"],
                future_points=10,
                image_contract=image_contract,
            ),
            out / "native",
        )
    else:
        exported = export_native_frames(
            plan["branch"]["scene"],
            plan["branch"]["rollout"],
            plan["render_report"],
            out / "native",
            anchor_tick=branch["anchor_tick"],
            image_contract=image_contract,
        )
    caches = write_counterfactual_caches(
        context["context"],
        context["egos"],
        context["observations"],
        out / "caches",
        native_log_name=exported["native_log_name"],
        token=exported["token"],
        branch_files=required,
        admission_profile=branch["trajectory_evidence"].get("admission_profile", "strict_world95_v1"),
    )
    row, files = seal_admission(
        exported, caches, plan["branch"], plan["visual_admission"], out / "admission.json"
    )
    from driversi_agents.chainflow.data import verify_records

    validation_steps = verify_records([row])
    if validation_steps != 40:
        raise ValueError("Synthetic cache validation accounting differs")
    check_files(plan["files"])
    report = dict(
        kind="chainflow_admitted_synthetic_sample",
        row=row,
        files=files,
        scored_agent_steps=160,
        training_started=False,
        trainable=True,
        physical_history_frames=4,
        actual_future_points=10,
        score_horizon_seconds=4,
        actual_world_context_seconds=5,
    )
    if history_only:
        report.update(image_contract=image_contract, future_images_included=False,
                      image_counts=exported["image_counts"])
    write_json(out / "dataset.json", report)
    return report
