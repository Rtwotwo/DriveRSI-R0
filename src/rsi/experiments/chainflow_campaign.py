"""ChainFlow stage1 full-test feedback import and native synthetic training campaign."""

import csv
import math
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json, task_state
from driversi.runtime.budget import write_json


def import_baseline(source, config_path):
    from driversi_agents.chainflow.benchmark import evaluation_identity, read_index

    source, config_path = Path(source), Path(config_path)
    c = read_mapping(config_path)
    summary = read_mapping(source / "summary.json")
    index = read_index(c["test_index"], "navtest")
    expected = set(index["expected_tokens"])
    if (
        len(expected) != 12146
        or not summary.get("complete")
        or summary.get("succeeded") != 12146
        or summary.get("failed") != 0
    ):
        raise ValueError("Complete official ChainFlow baseline required")
    identity = evaluation_identity(
        dict(
            index=c["test_index"],
            checkpoint=c["checkpoint"],
            backbone=c["backbone"],
            shards=4,
            seed=0,
        )
    )
    identity.pop("shard")
    if identity != summary["identity"]:
        raise ValueError("Parent checkpoint or native stage1 evaluation identity differs")
    rows = []
    with (source / "per-scene.csv").open() as stream:
        for row in csv.DictReader(stream):
            value = float(row["score"])
            if (
                row["valid"] != "True"
                or row["error"]
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Failed or non-normalized native score")
            rows.append(
                dict(
                    token=row["token"],
                    source_log=row["log_name"],
                    metrics={
                        "PDMS": value,
                        **{k: float(row[k]) for k in summary["metrics"] if k != "score"},
                    },
                )
            )
    if len(rows) != len(expected) or {x["token"] for x in rows} != expected:
        raise ValueError("Full benchmark coverage or unique token contract failed")
    mean = sum(x["metrics"]["PDMS"] for x in rows) / len(rows)
    if abs(mean - summary["metrics"]["score"]) > 1e-10 or abs(100 * mean - summary["pdms"]) > 1e-8:
        raise ValueError("Native score unit or aggregate differs")
    known = {x["token"]: x["source_log"] for x in index["records"]}
    if any(known[x["token"]] != x["source_log"] for x in rows):
        raise ValueError("Evaluation source log differs from official index")
    out = Path(c["output"]) / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    frozen_json(
        out / "plan.json",
        dict(
            kind="native_chainflow_benchmark_plan",
            records=index["records"],
            model="chainflow_stage1",
            expected=12146,
        ),
    )
    files = {
        str(p.resolve()): digest(p)
        for p in (
            source / "summary.json",
            source / "per-scene.csv",
            Path(c["test_index"]),
            Path(c["checkpoint"]),
            Path(c["backbone"]),
        )
    }
    frozen_json(
        out / "report.json",
        dict(
            kind="normalized_full_native_evaluation",
            model="chainflow_stage1",
            complete=True,
            completed=len(rows),
            expected=12146,
            rows=rows,
            metrics={"PDMS": mean},
            plan_sha256=digest(out / "plan.json"),
            files=files,
            score_unit="0_to_1",
            native_score_unmodified=True,
            adaptive_evaluation=True,
        ),
    )
    result = dict(
        status="verified",
        model="chainflow_stage1",
        completed=len(rows),
        pdms=100 * mean,
        zero=sum(x["metrics"]["PDMS"] == 0 for x in rows),
        nonzero_below_06=sum(0 < x["metrics"]["PDMS"] < 0.6 for x in rows),
        below_095=sum(x["metrics"]["PDMS"] < 0.95 for x in rows),
        tto=False,
        dit=False,
        source_summary=str(source / "summary.json"),
    )
    write_json(Path(c["output"]) / "baseline-import.json", result)
    return result


def process(generation, config, output, *, execution_gpu=None, quarantine=False, native_runner=None):
    from driversi.rendering.diagnostics.numeric_admission import admit
    from driversi.rsi.experiments.reactive_postprocess import run_native

    g, c = read_mapping(generation), read_mapping(config)
    from driversi.data.datasets.postprocessing import mode, training_only
    from driversi.data.datasets.synthetic_publish import dataset_root
    processing_mode = mode(c, production=not quarantine)
    if processing_mode == "training_only":
        if quarantine or not training_only(dataset_root().parent):
            raise ValueError("Training-only mode requires its active production dataset policy")
    elif not quarantine and training_only(dataset_root().parent):
        raise ValueError("Retrieval processing is deferred for this campaign")
    gpu = c["production_gpu"] if execution_gpu is None else execution_gpu
    if type(gpu) is not int or not 0 <= gpu <= 7:
        raise ValueError("Invalid native postprocessing GPU")
    out = Path(output)
    if quarantine:
        from driversi.agents.backends.native import project_root

        exp_root = (project_root() / "exp").resolve()
        isolated = Path(c["output"]).resolve()
        expected_database = isolated / "features.sqlite"
        marker = isolated / "quarantine.json"
        if (
            isolated == exp_root
            or not isolated.is_relative_to(exp_root)
            or not out.resolve().is_relative_to(isolated)
            or Path(c["generated_catalog"]).resolve() != expected_database
            or not marker.exists()
            or read_mapping(marker)
            != dict(
                kind="isolated_postprocess_benchmark",
                generated_catalog=str(expected_database),
                production_training=False,
            )
        ):
            raise ValueError("Benchmark requires its dedicated quarantined feature database")
    out.mkdir(parents=True, exist_ok=True)
    frozen_json(
        out / "inputs.json",
        dict(
            generation_sha256=digest(generation),
            config_sha256=digest(config),
            **({"execution_gpu": gpu} if execution_gpu is not None else {}),
            **({"quarantine": True} if quarantine else {}),
            **({"postprocessing_policy_sha256": digest(dataset_root().parent / "postprocessing.json")}
               if processing_mode == "training_only" else {}),
        ),
    )
    visual = admit(
        generation,
        Path(generation).parent.parent / "diversity.json",
        out / "visual",
        profile=c.get("visual_quality_profile", "experimental_numeric25_v1"),
    )
    if not visual["admitted"]:
        result = dict(
            status="visual_preflight_rejected",
            trainable=False,
            blockers=visual["rejection_reasons"],
        )
        write_json(out / "report.json", result)
        return result
    branch = read_mapping(g["branch_report"])
    execution = Path(branch["execution"])
    context = execution / "teacher-counterfactual-context.pkl"
    if str(context) not in branch["files"]:
        raise ValueError("Counterfactual scoring context required")
    files = {
        **visual["files"],
        str(out / "visual/admission.json"): digest(out / "visual/admission.json"),
    }
    plan = dict(
        output=str(out / "native"),
        branch=dict(
            report=g["branch_report"],
            scene=str(execution / "scene.npz"),
            rollout=str(execution / "teacher.npz"),
            trajectory_window=visual["trajectory_window"],
        ),
        render_report=g["render_payload"],
        context=str(context),
        visual_admission=str(out / "visual/admission.json"),
        maps_root=c["maps_root"],
        files=files,
        canonical_dataset=not quarantine,
    )
    (out / "native").mkdir(exist_ok=True)
    (native_runner or run_native)(
        "chainflow_export",
        dict(plan=plan),
        out / "calls/export",
        gpu=gpu,
        python=c["chainflow_python"],
        timeout=1800,
    )
    sample = out / "native/dataset.json"
    export = out / "native/native/export.json"
    if processing_mode == "training_only":
        from driversi.data.datasets.training_publish import index_training_package, DEFERRED
        reference = index_training_package(sample, export)
        result = dict(status="training_sample_ready", trainable=True,
                      model="chainflow_stage1", sample_report=str(sample),
                      feature_report=None, scene_classification=None,
                      postprocessing_mode=processing_mode, postprocessing=DEFERRED,
                      dataset_reference=reference, generation=str(Path(generation).resolve()),
                      images_changed=True, original_anchor_relabel=False,
                      language_supervision_required=False,
                      files={str(p): digest(p) for p in
                             (sample, export.resolve(), out / "visual/admission.json", Path(reference["path"]))})
        write_json(out / "report.json", result)
        return result
    (native_runner or run_native)(
        "features",
        dict(
            export_path=str(export),
            source_catalog=c["train_catalog"],
            profile_id=c["profile"],
            database=c["generated_catalog"],
            output=str(out / "features"),
            sample_report=str(sample),
        ),
        out / "calls/features",
        gpu=gpu,
        python=c["feature_python"],
    )
    feature = read_mapping(out / "features/report.json")
    if not feature["trainable"] or feature["parent_vectors_copied"]:
        raise ValueError("Fresh admitted feature required")
    reference = read_mapping(out / "calls/features/result.json").get("dataset_reference")
    if quarantine and reference is not None:
        raise ValueError("Benchmark unexpectedly published a canonical admission")
    if not quarantine and reference is None:
        raise ValueError("Production requires canonical dataset admission")
    result = dict(
        status="benchmark_sample_verified" if quarantine else "training_sample_ready",
        trainable=not quarantine,
        model="chainflow_stage1",
        sample_report=str(sample),
        feature_report=str(out / "features/report.json"),
        dataset_reference=reference,
        scene_classification=read_mapping(out / "calls/features/result.json").get("scene_classification"),
        generation=str(Path(generation).resolve()),
        images_changed=True,
        original_anchor_relabel=False,
        language_supervision_required=False,
        files={
            str(p): digest(p)
            for p in (sample, out / "features/report.json", out / "visual/admission.json")
        },
    )
    write_json(out / "report.json", result)
    return result


def target_stop_guard(path):
    """Settled local non-improvement stops before costly full benchmark nodes."""
    path = Path(path)
    if path.exists() and read_mapping(path).get('target_success') is False:
        raise RuntimeError('local_target_still_zero')


def train_pair(config, samples, output, *, replay_reports=()):
    from driversi.rsi.plugins.chainflow_synthetic import compile_paired_task
    from driversi_agents.chainflow.benchmark import make_manifest

    c = read_mapping(config)
    population = c.get('training_population', 'mixed')
    if population == 'synthetic_only':
        from driversi.rsi.decisions.problem_learning import source_quota
        from driversi_agents.chainflow.synthetic import validate_admission
        if replay_reports:
            raise ValueError('This problem trains on its new samples only')
        required_profile = c.get('visual_quality_profile')
        from driversi.rendering.diagnostics.numeric_admission import strict_problem_profile
        if not strict_problem_profile(required_profile,c.get('strict_source_psnr_gt')):
            raise ValueError('New-only problem training requires its declared strict visual policy')
        origins = []
        for path in samples:
            sample = read_mapping(path)
            proof = validate_admission(sample['row'], sample['files'])
            visual = read_mapping(proof['visual_admission']['path'])
            if visual.get('visual_quality_profile') != required_profile:
                raise ValueError('Legacy visual admission cannot enter a different problem profile')
            origins.append((proof.get('source_train_membership') or proof['parent'])['token'])
        if c['trajectory_family'].get('cohort_mode')=='global_50_min1_v1':
            quota=dict(passed=len(origins)==50 and set(origins)==set(c['required_source_tokens']))
        else:
            quota = source_quota(c['required_source_tokens'], origins,
                                 c['trajectory_family']['target_per_anchor'])
        if len(set(samples)) != len(samples) or not quota['passed']:
            raise ValueError('Every selected source must supply its own complete admitted quota')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if c.get("trajectory_family"):
        from driversi.rsi.experiments.reactive_cohort import family_training_gate

        summary = read_mapping(out.parent / "trajectory-family-summary.json")
        if c.get('training_family_summary'):
            ref=c['training_family_summary']
            if digest(ref['path'])!=ref['sha256']:raise ValueError('Training cohort changed')
            summary=read_mapping(ref['path'])
            for path,sha in summary['evidence'].items():
                if digest(path)!=sha:raise ValueError('Cumulative cohort evidence changed')
        plan = read_mapping(out.parent / "trajectory-family-plan.json")
        anchors = list(dict.fromkeys(a["anchor_assignment_id"] for a in plan["assignments"]))
        gate = family_training_gate(summary, anchors, c["trajectory_family"]["target_per_anchor"],
            **({'global_target':50} if c['trajectory_family'].get('cohort_mode')=='global_50_min1_v1' else {}))
        write_json(out / "per-anchor-training-gate.json", gate)
        if (
            not gate["passed"]
            or len(set(samples)) != len(samples)
            or set(samples) != set(summary["sample_reports"])
        ):
            raise ValueError(
                "Each source must have its complete qualified sample family before training"
            )
    manifest = out / "base-manifest.json"
    if not manifest.exists():
        make_manifest(c["train_index"], c["test_index"], manifest)
    # Two GPUs x batch two => every eight global samples exposes one new sample.
    updates = max(c.get("pair_optimizer_steps", 8), 2 * (len(samples) + len(replay_reports)))
    if population == 'synthetic_only':
        from driversi.rsi.decisions.problem_learning import sft_updates
        model = read_mapping(c['training_model_config'])
        world = len(model['settings']['distributed_training']['devices'])
        updates = sft_updates(len(samples), c['sft_epochs'], 2*world)
    if updates % 2:
        updates += 1
    ceiling = c.get("max_pair_optimizer_steps", 64)
    if type(ceiling) is not int or not 8 <= ceiling <= 256 or updates > ceiling:
        raise ValueError("Pair exceeds the explicit bounded update budget")
    task = out / "paired/task.json"
    capability_plan=out.parent/'trajectory-family-plan.json'
    if c.get('training_capability_plan'):
        ref=c['training_capability_plan']
        if digest(ref['path'])!=ref['sha256']:raise ValueError('Training capability plan changed')
        capability_plan=Path(ref['path'])
    if not task.exists():
        compile_paired_task(
            manifest,
            samples,
            c["training_model_config"],
            out / "paired",
            updates_per_arm=updates,
            epoch_training=True,
            pre_evaluation_gate=True,
            batch_size=2,
            capability_plan=capability_plan,
            skill_memory=out.parent / "generation-memory.sqlite",
            replay_reports=replay_reports,
            training_population=population,
            target_token=c.get('target_query_token'),
        )
    target_report = out/'tasks'/read_mapping(task)['id']/'treatment_pre_evaluation/target-feedback.json'
    from functools import partial
    guard = partial(target_stop_guard,target_report) if c.get('target_query_token') else None
    kwargs = dict(evidence_guard=guard) if guard else {}
    result = task_state(out / "harness.sqlite", read_mapping(task), task, out / "tasks", execute=True, **kwargs)
    local_failure = result.get('failure_reason') == 'local_target_still_zero'
    write_json(
        out / "training-status.json",
        dict(
            status='target_still_zero' if local_failure else result["status"],
            directory=result["directory"],
            model="chainflow_stage1",
            full_navtest_per_arm=12146,
            updates_per_arm=updates,
            qualified_samples=len(samples),
            target_feedback=str(target_report) if target_report.exists() else None,
        ),
    )
    return result["status"] == "completed"
