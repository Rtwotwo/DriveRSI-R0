"""Sealed physical branch and independent visual evidence to native training data."""

from pathlib import Path

from driversi.agents.backends.native import digest, path_at, project_root
from driversi.configuration import read_mapping
from driversi.rsi.contracts import _keys
from driversi.rsi.plugins.synthesis import check_files, protect, run_process
from driversi.rsi.workflow.results import result
from driversi.runtime.budget import write_json


def prepare(config, base, registry):
    _keys(config, ("python", "maps_root", "max_wall_seconds"), "ChainFlow synthetic export")
    if type(config["max_wall_seconds"]) is not int or not 30 <= config["max_wall_seconds"] <= 600:
        raise ValueError("Bounded native export budget required")
    root = project_root()
    sources = [
        root / name
        for name in (
            "agents/chainflow/synthetic_export.py",
            "agents/chainflow/synthetic.py",
            "agents/chainflow/data.py",
            "src/rendering/diagnostics/worldengine_admission.py",
            "src/evaluation/metrics/counterfactual_pdm.py",
            "src/rsi/correction/trajectory_admission.py",
            "src/data/datasets/navsim_synthetic.py",
        )
    ]
    return {
        **config,
        "python": str(path_at(base, config["python"])),
        "maps_root": str(path_at(base, config["maps_root"])),
        "files": {str(p): digest(p) for p in sources},
    }, 160


def execute(config, inputs, output, registry):
    from driversi.rendering.diagnostics.worldengine_admission import validate_visual_admission
    from driversi.rsi.plugins.native_branch import validate_trajectory_evidence

    branch = read_mapping(inputs["branch"]["artifacts"]["report"])
    rendered = read_mapping(inputs["rendered"]["artifacts"]["report"])
    visual = inputs["visual"]["artifacts"]["report"]
    validate_trajectory_evidence(branch["trajectory_evidence"])
    if not validate_visual_admission(visual)["admitted"]:
        raise ValueError("Independent visual admission rejected this branch")
    protect([branch["source_log"]], registry)
    window = inputs["branch"]["artifacts"]["trajectory_window"]
    if len(read_mapping(window)["future"]) != 10:
        raise ValueError("ChainFlow requires ten actual future points")
    identity = inputs["branch"]["summary"]["artifact_reference"]["identity"]
    if rendered["branch_identity"] != identity:
        raise ValueError("Rendering belongs to another executed branch")
    files = {
        **config["files"],
        **branch["files"],
        **branch["upstream_source_files"],
        **rendered["payload_files"],
    }
    for path in (
        inputs["branch"]["artifacts"]["report"],
        window,
        visual,
        inputs["rendered"]["artifacts"]["render_report"],
    ):
        files[path] = digest(path)
    check_files(files)
    execution = Path(branch["execution"])
    context = execution / "teacher-counterfactual-context.pkl"
    if str(context) not in branch["files"]:
        raise ValueError("Native training context lacks producer evidence")
    output.mkdir(parents=True)
    plan = dict(
        worker="driversi_agents.chainflow.synthetic_export",
        output=str(output),
        branch=dict(
            report=inputs["branch"]["artifacts"]["report"],
            scene=str(execution / "scene.npz"),
            rollout=str(execution / (branch["branch"] + ".npz")),
            trajectory_window=window,
        ),
        render_report=inputs["rendered"]["artifacts"]["render_report"],
        context=str(context),
        visual_admission=visual,
        maps_root=config["maps_root"],
        files=files,
    )
    write_json(output / "export-plan.json", plan)
    run_process(
        [
            config["python"],
            "-B",
            str(project_root() / "src/training/orchestration/worker.py"),
            str(output / "export-plan.json"),
        ],
        output,
        config["max_wall_seconds"],
    )
    check_files(files)
    report = read_mapping(output / "dataset.json")
    from driversi_agents.chainflow.synthetic import validate_admission

    validate_admission(report["row"], report["files"])
    return result(
        output,
        report,
        steps=report["scored_agent_steps"],
        artifacts={
            "dataset": str(output / "dataset.json"),
            "admission": str(output / "admission.json"),
        },
    )


_REVIEWED_KINDS = {
    "branch": "selected_native_branch",
    "rendered": "worldengine_inspection_window",
    "source": "worldengine_source_comparison",
}


def _stored_stage(reference, kind, registry):
    """Resolve only a completed typed receipt in this Harness ledger."""
    from driversi.rsi.workflow.artifacts import verify
    from driversi.rsi.workflow.store import HarnessStore

    _keys(reference, ("source_task", "source_stage"), "stored synthetic input")
    if any(not isinstance(reference[key], str) or not reference[key] for key in reference):
        raise ValueError("Explicit stored task and stage identities required")
    state = HarnessStore(registry).get(reference["source_task"])
    declaration = next(
        (row for row in state["task"]["steps"] if row["id"] == reference["source_stage"]), None
    )
    stage = next((row for row in state["stages"] if row["id"] == reference["source_stage"]), None)
    if (
        not declaration
        or not stage
        or declaration["output_kind"] != kind
        or stage["status"] != "completed"
    ):
        raise ValueError("Completed stored " + kind + " stage required")
    original = stage["result"]
    if original["status"] != "completed":
        raise ValueError("Stored stage receipt is not completed")
    ref = verify(original, state["id"], declaration)
    protect(state["task"]["source_groups"], registry)
    return state, original, ref


def prepare_reviewed(config, base, registry):
    """Freeze past artifacts after their exact pixels have been reviewed."""
    from driversi.rsi.plugins.worldengine import prepare_admission

    _keys(
        {k: v for k, v in config.items() if k != "visual_quality_profile"},
        (
            "branch",
            "rendered",
            "source",
            "visual_review",
            "python",
            "maps_root",
            "max_wall_seconds",
        ),
        "reviewed synthetic export",
    )
    native, cap = prepare(
        {key: config[key] for key in ("python", "maps_root", "max_wall_seconds")}, base, registry
    )
    admission, _ = prepare_admission(
        {
            "visual_review": config["visual_review"],
            "visual_quality_profile": config.get("visual_quality_profile", "source27_v1"),
        },
        base,
        registry,
    )
    references, groups = {}, set()
    for port, kind in _REVIEWED_KINDS.items():
        state, _, ref = _stored_stage(config[port], kind, registry)
        references[port] = {**config[port], "identity": ref["identity"]}
        groups.update(state["task"]["source_groups"])
    native["files"].update(
        {
            str(project_root() / path): digest(project_root() / path)
            for path in (
                "src/rsi/plugins/chainflow_synthetic.py",
                "src/rsi/plugins/worldengine.py",
                "src/rsi/workflow/artifacts.py",
                "src/rsi/workflow/store.py",
            )
        }
    )
    return {
        "native": native,
        "admission": admission,
        "references": references,
        "source_groups": sorted(groups),
    }, cap


def execute_reviewed(config, inputs, output, registry):
    """Reuse original identities, independently admit, then export if qualified."""
    from driversi.rsi.plugins.worldengine import execute_admission

    if inputs:
        raise ValueError("Reviewed export accepts only sealed stored references")
    check_files({**config["native"]["files"], **config["admission"]["files"]})
    originals, refs, groups = {}, {}, set()
    for port, kind in _REVIEWED_KINDS.items():
        item = config["references"][port]
        state, original, ref = _stored_stage(
            {key: item[key] for key in ("source_task", "source_stage")}, kind, registry
        )
        if ref["identity"] != item["identity"]:
            raise ValueError("Stored synthetic input changed after registration: " + port)
        originals[port], refs[port] = original, ref
        groups.update(state["task"]["source_groups"])
    if sorted(groups) != config["source_groups"]:
        raise ValueError("Stored synthetic source provenance changed")
    if not output.resolve().is_relative_to(project_root() / "exp"):
        raise ValueError("Reviewed export outputs must live under exp")
    output.mkdir(parents=True)
    write_json(
        output / "origin.json",
        {"references": refs, "rendering_reexecuted": False, "source_groups": sorted(groups)},
    )
    visual = execute_admission(
        config["admission"],
        {
            "branch": originals["branch"],
            "render": originals["rendered"],
            "source": originals["source"],
        },
        output / "visual",
        registry,
    )
    evidence = read_mapping(visual["artifacts"]["report"])
    if visual["status"] != "completed" or evidence.get("admitted") is not True:
        return result(
            output,
            {
                "kind": "chainflow_synthetic_visual_rejection",
                "trainable": False,
                "training_started": False,
                "rendering_reexecuted": False,
                "rejection_reasons": evidence.get(
                    "rejection_reasons", ["independent_visual_admission_rejected"]
                ),
                "source_references": {key: ref["identity"] for key, ref in refs.items()},
                "visual_evidence": evidence,
                "scored_agent_steps": 0,
            },
            status="rejected",
            reason="independent_visual_admission_rejected",
            artifacts={
                "visual_report": visual["artifacts"]["report"],
                "origin": str(output / "origin.json"),
            },
        )
    exported = execute(
        config["native"],
        {
            "branch": originals["branch"],
            "rendered": originals["rendered"],
            "visual": visual,
        },
        output / "export",
        registry,
    )
    check_files({**config["native"]["files"], **config["admission"]["files"]})
    report = {
        **read_mapping(exported["artifacts"]["report"]),
        "source_references": {key: ref["identity"] for key, ref in refs.items()},
        "rendering_reexecuted": False,
    }
    return result(
        output,
        report,
        status=exported["status"],
        reason=exported["reason"],
        steps=exported["sim_agent_steps"],
        artifacts={
            **{key: value for key, value in exported["artifacts"].items() if key != "report"},
            "visual_report": visual["artifacts"]["report"],
            "origin": str(output / "origin.json"),
        },
    )


def register(registry, Provider):
    registry.register(
        Provider(
            "chainflow_reviewed_synthetic_dataset",
            "1",
            (),
            "chainflow_admitted_synthetic_sample",
            prepare_reviewed,
            execute_reviewed,
            ("report", "dataset", "admission", "visual_report", "origin"),
        )
    )
    registry.register(
        Provider(
            "chainflow_synthetic_dataset",
            "1",
            (
                ("branch", "selected_native_branch"),
                ("rendered", "worldengine_inspection_window"),
                ("visual", "worldengine_visual_admission"),
            ),
            "chainflow_admitted_synthetic_sample",
            prepare,
            execute,
            ("report", "dataset", "admission"),
        )
    )


def compile_paired_task(
    base_manifest,
    sample_report,
    model_config,
    output,
    *,
    updates_per_arm=8,
    capability_plan=None,
    skill_memory=None,
    epoch_training=False,
    batch_size=1,
    pre_evaluation_gate=False,
    replay_reports=(),
    training_population='mixed',
    target_token=None,
):
    """Materialize a genuine intervention using the existing training providers.

    Both arms are prepared only after synthetic admission. The original pool
    and independent development rows stay byte-for-byte identical; the added
    sample must enter the bounded optimizer cursor before feedback can credit it.
    """
    import copy
    import os
    from collections import Counter

    from driversi.training.orchestration.lifecycle import sources
    from driversi_agents.chainflow.benchmark import evaluator_files, sample_indices
    from driversi_agents.chainflow.data import validate_manifest
    from driversi_agents.chainflow.synthetic import validate_admission

    if type(epoch_training) is not bool:
        raise ValueError("Explicit boolean epoch_training required")
    if training_population not in ('mixed', 'synthetic_only'):
        raise ValueError('Unknown native training population')
    synthetic_only = training_population == 'synthetic_only'
    if synthetic_only and (not epoch_training or replay_reports):
        raise ValueError('Synthetic-only adaptation requires epoch training and no replay')
    if target_token is not None and (not epoch_training or not pre_evaluation_gate):
        raise ValueError('Explicit target test requires epoch training and exposure gate')
    base = validate_manifest(base_manifest)
    if bool(capability_plan) != bool(skill_memory):
        raise ValueError("Capability plan and skill memory must be configured together")
    benchmark = base["schema_version"] in (2, 3)
    pilot = base["schema_version"] == 3
    evaluation_access = (
        {
            "evaluation_protocol": base["benchmark"]["protocol"],
            **({"evaluation_subset": base["benchmark"]["test_subset"]} if pilot else {}),
        }
        if benchmark
        else {}
    )
    if (
        type(updates_per_arm) is not int
        or updates_per_arm < 2
        or updates_per_arm % 2
        or (not epoch_training and updates_per_arm > 128)
    ):
        raise ValueError("Even pilot or epoch updates required; >128 needs explicit epoch training")
    world_size = 1
    if epoch_training:
        from driversi_agents.chainflow.sampling import distributed_settings

        if not benchmark or type(batch_size) is not int or not 1 <= batch_size <= 64:
            raise ValueError(
                "Epoch pairs require complete navtrain and explicit per-rank batch size"
            )
        settings = distributed_settings(read_mapping(model_config)["settings"])
        world_size = len(settings["devices"])
        if settings["synthetic_every"] != (1 if synthetic_only else 8):
            raise ValueError("Sampling interval differs from declared training population")
    elif batch_size != 1:
        raise ValueError("Legacy paired recipe uses batch size one")
    global_batch = batch_size * world_size
    paths = list(sample_report) if isinstance(sample_report, (list, tuple)) else [sample_report]
    sample_limit = 128 if epoch_training else 64
    if not paths or len(paths) > sample_limit:
        raise ValueError(f"One to {sample_limit} admitted synthetic receipts required")
    samples = [read_mapping(path) for path in paths]
    replay_paths = list(replay_reports)
    if replay_paths and (not epoch_training or len(replay_paths) > 32):
        raise ValueError("Shared replay requires bounded native epoch training")
    replay = [read_mapping(path) for path in replay_paths]
    ids = {row["sample_id"] for rows in base["splits"].values() for row in rows}
    for sample in samples + replay:
        if sample.get("kind") != "chainflow_admitted_synthetic_sample":
            raise ValueError("Admitted synthetic sample receipt required")
        validate_admission(sample["row"], sample["files"])
        if sample["row"]["kind"] != "synthetic_navtrain":
            raise ValueError("Real synthetic data intervention required")
        identity = sample["row"]["sample_id"]
        if identity in ids:
            raise ValueError("Synthetic intervention duplicates an existing sample")
        ids.add(identity)
    if any(row["kind"] != "original_navtrain" for row in base["splits"]["train"]):
        raise ValueError("Continued-training control must retain original data")
    if not benchmark and len(base["splits"]["train"]) + len(samples) > updates_per_arm:
        raise ValueError("Pilot update budget cannot consume every original and synthetic sample")
    if benchmark and updates_per_arm * global_batch // (1 if synthetic_only else 8) < len(samples) + len(replay):
        raise ValueError("Bounded mixing budget must expose every admitted synthetic sample")
    out = Path(output).resolve()
    if not out.is_relative_to(project_root() / "exp") or out.exists():
        raise ValueError("New paired experiment must live under exp")
    model_path = Path(model_config).resolve(strict=True)
    model = read_mapping(model_path)
    skill_binding = None
    if capability_plan:
        from driversi.rsi.decisions.capability_skills import validate_plan

        plan_path = Path(capability_plan).resolve(strict=True)
        skill_plan = validate_plan(read_mapping(plan_path))
        if skill_plan["context"]["model"] != "chainflow_stage1":
            raise ValueError("Skill plan belongs to another model family")
        from driversi.rsi.decisions.capability_skills import validate_sample_lineage

        assignment_ids = set()
        for sample in samples:
            assignment_ids.update(
                validate_sample_lineage(skill_plan, [sample["row"]], sample["files"])
            )
        skill_binding = {
            "plan": str(plan_path),
            "plan_sha256": digest(plan_path),
            "plan_id": skill_plan["plan_id"],
            "assignment_ids": sorted(assignment_ids),
            "model": "chainflow_stage1",
            "parent_checkpoint": skill_plan["context"]["parent_checkpoint"],
            "evaluation_split": "official_navtest_locked_subset_pilot"
            if pilot
            else "official_navtest"
            if benchmark
            else "navtrain_engineering_regression",
            "added_sample_ids": [x["row"]["sample_id"] for x in samples],
        }
    recipe = model.get("settings", {}).get("training_recipe", {})
    if model.get("model") != "chainflow_stage1" or not {
        "scheduler_dataset_size",
        "scheduler_num_epochs",
    } <= set(recipe):
        raise ValueError("Explicit common native incremental schedule required")
    if any(
        type(recipe[k]) is not int or recipe[k] <= 0
        for k in ("scheduler_dataset_size", "scheduler_num_epochs")
    ):
        raise ValueError("Positive native schedule horizon required")
    if ((recipe["scheduler_dataset_size"] + global_batch - 1) // global_batch) * recipe[
        "scheduler_num_epochs"
    ] < updates_per_arm:
        raise ValueError("Pair update budget exceeds common native schedule horizon")
    arms = {"control": copy.deepcopy(base), "treatment": copy.deepcopy(base)}
    for dataset in arms.values():
        for sample in replay:
            dataset['splits']['train'].append(sample['row'])
            for path, sha in sample['files'].items():
                if path in dataset['files'] and dataset['files'][path] != sha:
                    raise ValueError('Shared replay input conflicts with original contents')
                dataset['files'][path] = sha
    for sample in samples:
        arms["treatment"]["splits"]["train"].append(sample["row"])
        for path, sha in sample["files"].items():
            if path in arms["treatment"]["files"] and arms["treatment"]["files"][path] != sha:
                raise ValueError("Synthetic input conflicts with control file contents")
            arms["treatment"]["files"][path] = sha
    # Confirm original-log isolation before creating requests or launching work.
    for dataset in arms.values():
        groups = {
            side: {"source_logs": sorted({x["source_log"] for x in dataset["splits"][side]})}
            for side in ("train", "development")
        }
        sources({**groups, "parent_provenance": "unknown", **evaluation_access})
    out.mkdir(parents=True)
    count = len(base["splits"]["development"])
    budget = dict(
        max_wall_seconds=2400 if pilot else 21600 if benchmark else 1200,
        max_optimizer_steps=updates_per_arm // 2,
        max_environment_steps=0,
        max_scoring_agent_steps=max(
            11000,
            2750 * global_batch * (updates_per_arm // 2) + 40 * len(samples) * world_size,
            count * 160 + 40 * len(samples) if benchmark else 0,
        ),
    )
    for side, dataset in arms.items():
        folder = out / side
        folder.mkdir()
        manifest = folder / "manifest.json"
        write_json(manifest, dataset)
        validate_manifest(manifest)
        data = {
            "manifest": str(manifest),
            "parent_provenance": "unknown",
            **evaluation_access,
            **{
                split: {"source_logs": sorted({x["source_log"] for x in dataset["splits"][split]})}
                for split in ("train", "development")
            },
        }
        write_json(folder / "data.json", data)
        write_json(
            folder / "request.json",
            dict(
                schema_version=4 if epoch_training else 3 if benchmark else 2,
                id=side + "-synthetic-pilot",
                model_config=str(model_path),
                mode="supervised",
                initialization="model_weights",
                data=str(folder / "data.json"),
                pilot=dict(seed=42, batch_size=batch_size, evaluation_steps=count),
                budget=budget,
            ),
        )
        write_json(
            folder / "evaluator.json",
            dict(
                schema_version=3 if pilot else 2 if benchmark else 1,
                model="chainflow_stage1",
                protocol="chainflow_native_navsim_pdm_v1",
                development_source_logs=data["development"]["source_logs"],
                dataset_manifest_sha256=digest(manifest),
                metric_cache=os.path.commonpath(
                    [
                        str(Path(row["metric_cache"]).resolve().parent)
                        for row in base["splits"]["development"]
                    ]
                ),
                files=(
                    evaluator_files(base)
                    if benchmark
                    else {
                        row["metric_cache"]: dataset["files"][row["metric_cache"]]
                        for row in dataset["splits"]["development"]
                    }
                ),
            ),
        )
    steps = []
    for side in ("control", "treatment"):
        request = str(out / side / "request.json")
        steps.extend(
            [
                dict(
                    id=side + "_train",
                    provider="native_model_unified_training",
                    inputs={},
                    config={"request": request},
                    max_wall_seconds=budget["max_wall_seconds"] + 50,
                ),
                dict(
                    id=side + "_resume",
                    provider="native_model_unified_resume",
                    inputs={"student": side + "_train"},
                    config={"request": request},
                    max_wall_seconds=budget["max_wall_seconds"] + 50,
                ),
                dict(
                    id=side + "_evaluation",
                    provider="native_model_unified_evaluation",
                    inputs={"student": side + "_resume"},
                    config={"request": request, "evaluator": str(out / side / "evaluator.json")},
                    max_wall_seconds=budget["max_wall_seconds"] + 50,
                ),
            ]
        )
    steps.extend(
        [
            dict(
                id="treatment_replay",
                provider="native_model_unified_resume",
                inputs={"student": "treatment_train"},
                config={"request": str(out / "treatment/request.json")},
                max_wall_seconds=budget["max_wall_seconds"] + 50,
            ),
            dict(
                id="treatment_feedback",
                provider="native_model_unified_feedback",
                inputs={
                    "student": "treatment_resume",
                    "replay": "treatment_replay",
                    "evaluation": "treatment_evaluation",
                },
                config={},
                max_wall_seconds=600 if benchmark else 60,
            ),
            dict(
                id="paired_feedback",
                provider="chainflow_paired_feedback",
                inputs={
                    "control_student": "control_resume",
                    "treatment_student": "treatment_resume",
                    "control_evaluation": "control_evaluation",
                    "treatment_evaluation": "treatment_evaluation",
                },
                config={"skill_binding": skill_binding} if skill_binding else {},
                max_wall_seconds=600 if benchmark else 60,
            ),
        ]
    )
    if pre_evaluation_gate:
        indexed = {x["id"]: x for x in steps}
        gates = []
        for side in ("control", "treatment"):
            gid = side + "_pre_evaluation"
            gates.append(
                dict(
                    id=gid,
                    provider="chainflow_pair_pre_evaluation_gate",
                    inputs=dict(
                        control="control_resume",
                        treatment="treatment_resume",
                        replay="treatment_replay",
                    ),
                    config=dict(side=side, **({'target_token': target_token}
                                if target_token is not None and side=='treatment' else {})),
                    max_wall_seconds=660 if target_token is not None and side=='treatment' else 60,
                )
            )
            indexed[side + "_evaluation"]["inputs"]["student"] = gid
            indexed["paired_feedback"]["inputs"][side + "_student"] = gid
        indexed["treatment_feedback"]["inputs"]["student"] = "treatment_pre_evaluation"
        steps = (
            [
                indexed[k]
                for k in (
                    "control_train",
                    "control_resume",
                    "treatment_train",
                    "treatment_resume",
                    "treatment_replay",
                )
            ]
            + gates
            + [
                indexed[k]
                for k in (
                    "control_evaluation",
                    "treatment_evaluation",
                    "treatment_feedback",
                    "paired_feedback",
                )
            ]
        )
    if skill_binding:
        steps.append(
            dict(
                id="skill_memory",
                provider="capability_skill_feedback",
                inputs={"feedback": "paired_feedback"},
                config={
                    "plan": str(Path(capability_plan).resolve()),
                    "memory": str(Path(skill_memory).resolve()),
                },
                max_wall_seconds=60,
            )
        )
    task = dict(
        schema_version=2,
        id=out.name,
        label="Admitted new-history synthetic intervention versus equal-budget original continued training",
        budget=dict(
            max_wall_seconds=7 * (budget["max_wall_seconds"] + 50) + 1200,
            max_sim_agent_steps=7 * budget["max_scoring_agent_steps"],
        ),
        steps=steps,
    )
    write_json(out / "task.json", task)
    expected_epoch_exposures = {}
    if epoch_training:
        from driversi_agents.chainflow.sampling import GlobalEpochCursor

        for side, dataset in arms.items():
            cursor = GlobalEpochCursor(dataset["splits"]["train"], 42, batch_size, world_size,
                                       1 if synthetic_only else 8)
            counts = Counter()
            for _ in range(updates_per_arm):
                ids, _ = cursor.next_batch()
                counts.update(dataset["splits"]["train"][i]["sample_id"] for i in ids)
            expected_epoch_exposures[side] = counts
    write_json(
        out / "intervention.json",
        dict(
            base_manifest_sha256=digest(base_manifest),
            sample_reports={str(Path(p).resolve()): digest(p) for p in paths},
            model_config_sha256=digest(model_path),
            added_sample_ids=[x["row"]["sample_id"] for x in samples],
            shared_replay_sample_ids=[x['row']['sample_id'] for x in replay],
            replay_reports={str(Path(p).resolve()): digest(p) for p in replay_paths},
            optimizer_updates_per_arm=updates_per_arm,
            extra_resume_verification_updates=updates_per_arm // 2,
            sampling_protocol="synthetic_only_epochs_original_only_control"
            if synthetic_only else "global_shuffled_epoch_synthetic_insertions"
            if epoch_training
            else "seeded_full_navtrain_pool_synthetic_every_8"
            if benchmark
            else "sequential_pilot",
            epoch_backend=epoch_training,
            world_size=world_size,
            global_batch_size=global_batch,
            full_epoch_training=False,
            evaluation_split="official_navtest_locked_subset_pilot"
            if pilot
            else "official_navtest"
            if benchmark
            else "navtrain_engineering_regression",
            full_benchmark_evaluation=benchmark and not pilot,
            original_training_pool=len(base["splits"]["train"]),
            expected_sample_exposures=expected_epoch_exposures
            if epoch_training
            else (
                {
                    side: Counter(
                        dataset["splits"]["train"][i]["sample_id"]
                        for i in sample_indices(dataset["splits"]["train"], updates_per_arm, 42)
                    )
                    for side, dataset in arms.items()
                }
                if benchmark
                else {
                    side: {
                        row["sample_id"]: sum(
                            i % len(dataset["splits"]["train"]) == j for i in range(updates_per_arm)
                        )
                        for j, row in enumerate(dataset["splits"]["train"])
                    }
                    for side, dataset in arms.items()
                }
            ),
            data_expansion_semantics="frozen_batch_not_automatic_model_promotion",
            development_samples=count,
            training_started=False,
            training_population=training_population,
            original_pool_role='provenance_only_in_treatment' if synthetic_only else 'sampled',
        ),
    )
    return out / "task.json"
