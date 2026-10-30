"""Bounded four-camera GS window through existing typed DriveHarness providers.

This command produces sealed render and canonical-source evidence. It never
grants visual admission; an exact-pixel review must run before dataset export.
"""

import argparse
import ast
import json
from pathlib import Path


def interpreter_path(value):
    # A virtualenv executable is commonly a symlink; resolving it loses its
    # environment selection even though the binary bytes are the same.
    path = Path(value).absolute()
    if not path.is_file():
        raise ValueError("Python executable is missing")
    return path


def camera_contract(config_path, feature_builder_path):
    """Read enabled sensors and their actual native image-stack order."""
    import yaml

    config = yaml.safe_load(Path(config_path).read_text())["config"]
    enabled = {key for key, value in config.items() if key.startswith("cam_") and value}
    tree = ast.parse(Path(feature_builder_path).read_text())
    orders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "cameras" for t in node.targets):
            continue
        values = node.value.elts
        if values and all(
            isinstance(x, ast.Attribute) and x.attr.startswith("cam_") for x in values
        ):
            orders.append([x.attr for x in values])
    if len(orders) != 1 or not enabled <= set(orders[0]):
        raise ValueError("Unambiguous native feature-builder camera order required")
    order = [key.upper() for key in orders[0] if key in enabled]
    required = {"CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"}
    if set(order) != required or len(order) != 4:
        raise ValueError("Native four-camera stage1 configuration required")
    if any(config[key] != [3] for key in enabled):
        raise ValueError("This pilot preserves the native current-camera stage1 input")
    if config.get("num_poses") != 8 or config.get("long_trajectory_additional_poses") != 2:
        raise ValueError("Native ten-future trajectory target required")
    return {
        "native_stack_order": order,
        "export_camera_set": sorted(required),
        "native_image_time_index": 3,
        "history_ticks": [5, 10, 15, 20],
        "future_ticks": list(range(25, 71, 5)),
        "history_rgb_usage": "all_four_archived_native_stage1_consumes_current_four_camera_images",
        "future_rgb_usage": "supervision_archive_never_current_policy_input",
    }


def task_spec(task_id, source_task, source_stage, asset_root, python, gpu):
    if type(gpu) is not int or gpu < 0:
        raise ValueError("Explicit GPU required")
    asset_root = Path(asset_root)
    return {
        "schema_version": 2,
        "id": task_id,
        "label": "Same sealed physical branch: full native four-camera history and future GS fallback",
        "budget": {"max_wall_seconds": 1500, "max_sim_agent_steps": 0},
        "steps": [
            {
                "id": "reuse",
                "provider": "completed_navsim_teacher_reference",
                "inputs": {},
                "config": {"source_task": source_task, "source_stage": source_stage},
                "max_wall_seconds": 30,
            },
            {
                "id": "select",
                "provider": "native_branch_selection",
                "inputs": {"world": "reuse"},
                "config": {"branch": "teacher", "future_frames": 10},
                "max_wall_seconds": 30,
            },
            {
                "id": "render",
                "provider": "mtgs_selected_branch_render",
                "inputs": {"branch": "select"},
                "config": {
                    "config_path": str(asset_root / "config.yml"),
                    "checkpoint_path": str(asset_root / "step-000030000.ckpt"),
                    "road_block_config": str(asset_root / "road_block.yml"),
                    "travel_id": 2,
                    "python": str(python),
                    "gpu": gpu,
                    "cuda_home": "/usr/local/cuda-12.4",
                    "max_wall_seconds": 600,
                },
                "max_wall_seconds": 630,
            },
            {
                "id": "source",
                "provider": "worldengine_source_reconstruction",
                "inputs": {"render": "render"},
                "config": {
                    "python": str(python),
                    "gpu": gpu,
                    "cuda_home": "/usr/local/cuda-12.4",
                    "max_wall_seconds": 600,
                },
                "max_wall_seconds": 630,
            },
        ],
    }


def run(args):
    from driversi.agents.backends.native import digest, project_root
    from driversi.rsi.workflow.engine import register
    from driversi.rsi.workflow.engine import run as execute

    root = project_root()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "exp"):
        raise ValueError("Use a fresh exp output directory; failed attempts remain preserved")
    config = (
        root
        / "agents/chainflow/native/navsim/planning/script/config/common/agent/chainflow_vla_stage1.yaml"
    )
    builder = (
        root / "agents/chainflow/native/navsim/agents/chainflow_vla/builders/feature_builder.py"
    )
    contract = camera_contract(config, builder)
    output.mkdir(parents=True)
    task = task_spec(
        args.task_id,
        args.source_task,
        args.source_stage,
        args.asset_root.resolve(),
        interpreter_path(args.python),
        args.gpu,
    )
    task_path = output / "task.json"
    task_path.write_text(json.dumps(task, indent=2))
    receipt = {
        "kind": "bounded_multiview_gs_window",
        "status": "prepared",
        "trainable": False,
        "camera_contract": contract,
        "native_camera_sources": {str(p): digest(p) for p in (config, builder)},
        "registry": str(args.registry.resolve()),
        "task": str(task_path),
        "fallback_reason": "Native video assistance lacks causal history and joint multiview certification",
        "visual_admission": "independent_exact_pixel_review_required",
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(receipt, indent=2))
    if args.prepare_only:
        return receipt
    state = register(args.registry.resolve(), task_path, output / "tasks")
    state = execute(args.registry.resolve(), state["id"])
    (output / "task-result.json").write_text(json.dumps(state, indent=2))
    receipt.update(status=state["status"], failure_reason=state.get("failure_reason"))
    receipt["stages"] = {
        row["id"]: row.get("result", {}).get("artifacts", {}) for row in state["stages"]
    }
    report_path.write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registry", "asset-root", "python", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("task-id", "source-task", "source-stage"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--prepare-only", action="store_true")
    outcome = run(parser.parse_args())
    print(json.dumps(outcome, indent=2))
    raise SystemExit(0 if outcome["status"] in {"prepared", "completed"} else 1)


if __name__ == "__main__":
    main()
