"""Immutable physical rollout -> dynamic history RGB -> native training sample.

The caller is the persistent Harness controller. Completed rendering and export
receipts are reconciled, never silently repeated after an unknown execution.
"""

from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.rsi.plugins.synthesis import check_files
from driversi.runtime.budget import write_json

CONTRACT = "history_rgb_future_states_v1"
PROFILE = "dynamic_history25_v1"
CONTRACTS = {
    CONTRACT: (PROFILE, 16),
    "anchor_rgb_history_future_states_v1": ("dynamic_anchor25_v1", 4),
}


def settled_storage_failure(directory):
    """Only a finished native process with a specific storage errno may retry."""
    import os
    directory = Path(directory)
    exit_path, launch_path = directory / "exit.json", directory / "launch.json"
    if not exit_path.exists() or not launch_path.exists():
        return False
    receipt, launch = read_mapping(exit_path), read_mapping(launch_path)
    if receipt.get("returncode") in (None, 0, 124, 130):
        return False
    if receipt.get("job_sha256") != digest(directory / "job.json"):
        return False
    try:
        os.killpg(launch["pid"], 0)
        return False
    except ProcessLookupError:
        pass
    text = (directory / "worker.log").read_text(errors="replace")[-16000:]
    return any(mark in text for mark in ("[Errno 39] Directory not empty", "[Errno 116] Stale file handle"))


def _native_call(action, arguments, directory, **kwargs):
    from driversi.rsi.experiments.reactive_postprocess import run_native
    try:
        run_native(action, arguments, directory, **kwargs)
    except ValueError:
        if settled_storage_failure(directory):
            return False
        raise
    return True


def render_and_check(generation_report, diversity_report, bundle_root, output, image_contract=CONTRACT, gsplat_runtime_witness=None):
    import torch
    from driversi.contracts.scene import Scene
    from driversi.rendering.gaussian.worldengine_replay import run as render
    from driversi.rendering.gaussian.worldengine_source import run as compare_source
    from driversi.rendering.diagnostics.numeric_admission import admit, validate_diversity

    torch.cuda.set_per_process_memory_fraction(0.20, 0)
    runtime_files = []
    if gsplat_runtime_witness is not None:
        from driversi.rendering.gaussian.verified_gsplat_runtime import abi, validate_witness, load_extension
        witness = Path(gsplat_runtime_witness).resolve(strict=True)
        extension = validate_witness(read_mapping(witness), abi())
        load_extension(extension)
        runtime_files = [witness, extension]
    out, bundle = Path(output), Path(bundle_root)
    out.mkdir(parents=True, exist_ok=True)
    profile, _ = CONTRACTS[image_contract]
    g, _ = validate_diversity(generation_report, diversity_report, profile=profile)
    branch_path = Path(g["branch_report"])
    branch = read_mapping(branch_path)
    manifest_path = bundle / "dynamic-binding-manifest.json"
    manifest = read_mapping(manifest_path)
    checkpoint, metadata = bundle / "background" / (bundle.name + ".ckpt"), bundle / "video_scene_dict.pkl"
    if (manifest["checkpoint_sha256"] != digest(checkpoint)
        or manifest["metadata_sha256"] != digest(metadata)
        or manifest["source_log"] != branch["source_log"]
        or not manifest["object_nodes"]):
        raise ValueError("New dynamic asset lacks sealed actual object nodes")
    execution = Path(branch["execution"])
    scene_path, rollout_path = execution / "scene.npz", execution / (branch["branch"] + ".npz")
    scene = Scene.load(scene_path)
    asset_files = [checkpoint, metadata, manifest_path, bundle / "road_height_map/road_height_map.npy", bundle / "road_height_map/sim2.json", *runtime_files]
    rendered = out / "render/report.json"
    if not rendered.exists():
        render(scene_path, rollout_path, bundle, scene.metadata["navsim_source"]["log_path"], out / "render",
               future_points=10, image_contract=image_contract, width=960, asset_files=asset_files)
    payload = read_mapping(rendered)
    if payload.get("status") != "rendered_diagnostic" or payload.get("image_contract") != image_contract:
        raise ValueError("Existing render is incomplete; reconcile instead of overwriting")
    source_path = out / "source/report.json"
    if not source_path.exists():
        compare_source(rendered, out / "source")
    # Preserve the exact physical identity. Rendering does not create a new rollout.
    envelope_path = out / "render-envelope.json"
    old_envelope = read_mapping(g["render_report"])
    frozen_json(envelope_path, dict(branch_identity=old_envelope["branch_identity"],
                image_contract=image_contract, render_report=str(rendered), inputs=payload["inputs"],
                reused_physical_rollout=True, measured_new_simulation_steps=0))
    files = {str(p): digest(p) for p in (branch_path, rendered, source_path, envelope_path)}
    regenerated = dict(g, render_report=str(envelope_path), render_payload=str(rendered),
                       source_report=str(source_path), files=files, image_contract=image_contract,
                       reused_physical_rollout=True, status="rendered")
    generation = out / "generation.json"
    frozen_json(generation, regenerated)
    result = admit(generation, diversity_report, out / "visual", profile=profile)
    summary = dict(status="admitted" if result["admitted"] else "rejected",
                   reason=result.get("rejection_reasons", []), generation=str(generation),
                   admission=str(out / "visual/admission.json"), image_contract=image_contract,
                   mean_source_psnr_db=result["mean_source_psnr_db"], reused_physical_rollout=True,
                   files={str(p): digest(p) for p in (generation, out / "visual/admission.json", out / "visual/result.json")})
    write_json(out / "render-check.json", summary)
    return summary


def regenerate(generation_report, diversity_report, bundle_root, output, *, gpu, render_python, image_contract=CONTRACT, gsplat_runtime_witness=None):
    from driversi_agents.chainflow.synthetic import validate_admission

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    _, image_count = CONTRACTS[image_contract]
    job = dict(generation_report=str(Path(generation_report).resolve()), diversity_report=str(Path(diversity_report).resolve()),
               bundle_root=str(Path(bundle_root).resolve()), output=str(out / "processing"), image_contract=image_contract)
    runtime_seal = {}
    if gsplat_runtime_witness is not None:
        witness = str(Path(gsplat_runtime_witness).resolve(strict=True))
        job["gsplat_runtime_witness"] = witness
        runtime_seal = {"gsplat_runtime_witness_sha256": digest(witness)}
    frozen_json(out / "inputs.json", dict(**job, generation_sha256=digest(generation_report),
                diversity_sha256=digest(diversity_report), manifest_sha256=digest(Path(bundle_root) / "dynamic-binding-manifest.json"), **runtime_seal))
    result_path = out / "report.json"
    if result_path.exists():
        result = read_mapping(result_path)
        check_files(result.get("files", {}))
        if result["status"] == "training_sample_ready":
            sample = read_mapping(result["sample_report"])
            validate_admission(sample["row"], sample["files"])
        return result
    if not _native_call("dynamic_history_render", job, out / "calls/render", gpu=gpu, timeout=1800, python=render_python):
        result = dict(status="retryable", reason="known_transient_storage", result_known=True, no_inflight=True)
        write_json(result_path, result)
        return result
    checked = read_mapping(out / "processing/render-check.json")
    if checked["status"] != "admitted":
        result = dict(checked, status="rejected", trainable=False)
        write_json(result_path, result)
        return result
    g = read_mapping(checked["generation"])
    branch_path = Path(g["branch_report"])
    branch = read_mapping(branch_path)
    execution = Path(branch["execution"])
    context = execution / "teacher-counterfactual-context.pkl"
    if str(context) not in branch["files"]:
        raise ValueError("Sealed counterfactual scoring context required")
    visual = read_mapping(checked["admission"])
    plan = dict(output=str(out / "native"),
                branch=dict(report=str(branch_path), scene=str(execution / "scene.npz"),
                            rollout=str(execution / (branch["branch"] + ".npz")),
                            trajectory_window=str(branch_path.parent / "trajectory-window.json")),
                render_report=g["render_payload"], context=str(context), visual_admission=checked["admission"],
                maps_root=str(project_root().parent / "dataset/maps"), canonical_dataset=True,
                image_contract=image_contract, files={**visual["files"], checked["admission"]: digest(checked["admission"])})
    if not _native_call("chainflow_export", {"plan": plan}, out / "calls/export", gpu=gpu, timeout=600,
                        python=project_root() / ".venvs/augustus/bin/python"):
        result = dict(status="retryable", reason="known_transient_storage", result_known=True, no_inflight=True)
        write_json(result_path, result)
        return result
    sample_path = out / "native/dataset.json"
    sample = read_mapping(sample_path)
    validate_admission(sample["row"], sample["files"])
    from driversi.data.datasets.training_publish import index_training_package
    reference = index_training_package(sample_path, out / "native/native/export.json")
    result = dict(status="training_sample_ready", trainable=True, sample_report=str(sample_path),
                  image_contract=image_contract, future_rgb_count=0, historical_rgb_count=image_count,
                  reused_physical_rollout=True, mean_source_psnr_db=checked["mean_source_psnr_db"],
                  generation=g, dataset_reference=reference,
                  files={str(sample_path): digest(sample_path), reference["path"]: reference["sha256"], **checked["files"]})
    write_json(result_path, result)
    return result
