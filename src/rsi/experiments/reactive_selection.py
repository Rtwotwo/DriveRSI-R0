"""Complete private test feedback to independently sourced navval production."""

import math
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.data.catalog.store import Catalog
from driversi.data.catalog.vectors import VectorStore, normalize
from driversi.rsi.experiments.navval_batch_retrieval import retrieve
from driversi.rsi.experiments.navval_retrieval import navval_members
from driversi.runtime.budget import write_json


def ordered_failures(report, expected):
    rows = report.get("rows", [])
    tokens = [r["token"] for r in rows]
    if (
        not report.get("complete")
        or report.get("completed") != len(expected)
        or len(tokens) != len(set(tokens))
        or set(tokens) != set(expected)
    ):
        raise ValueError("Exact full benchmark required before adaptive mapping")
    for r in rows:
        score = r["metrics"]["PDMS"]
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid official PDMS")
    return sorted(
        [r for r in rows if r["metrics"]["PDMS"] < 0.95],
        key=lambda r: (r["metrics"]["PDMS"], r["token"]),
    )


def select(
    config_path, coverage_path, output, *, max_queries=24, max_candidates=16, preferred_bundles=()
):
    from driversi.rsi.experiments.reactive_generation import assignment_plan

    if not 1 <= max_queries <= 64 or not 1 <= max_candidates <= 16:
        raise ValueError("Bounded pilot query/candidate budget required")
    config = read_mapping(config_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    full = Path(config["output"]) / "baseline"
    plan = read_mapping(full / "plan.json")
    report = read_mapping(full / "report.json")
    if report["plan_sha256"] != digest(full / "plan.json") or len(plan["records"]) != 12146:
        raise ValueError("Parent full evaluation identity differs")
    failures = ordered_failures(report, [r["token"] for r in plan["records"]])
    frozen = dict(
        config_sha256=digest(config_path),
        coverage_sha256=digest(coverage_path),
        baseline_sha256=digest(full / "report.json"),
        max_queries=max_queries,
        max_candidates=max_candidates,
        preferred_bundles=list(preferred_bundles),
    )
    if (out / "selection.json").exists():
        old = read_mapping(out / "selection.json")
        if old["inputs"] != frozen:
            raise ValueError("Selection input changed")
        return old
    q = Catalog(config["test_catalog"], "navtest", readonly=True)
    t = Catalog(config["train_catalog"], "navtrain", readonly=True)
    try:
        profile = config["profile"]
        qs, ts = VectorStore(q), VectorStore(t)
        if qs.profile_data(profile) != ts.profile_data(profile):
            raise ValueError("Feature profiles differ")
        allowed, _ = navval_members(t)
        coverage = read_mapping(coverage_path)
        for p, sha in coverage["files"].items():
            if digest(p) != sha:
                raise ValueError("Asset coverage source changed")
        assets = sorted(
            coverage["rows"],
            key=lambda x: (
                x["bundle_root"] not in preferred_bundles,
                x["source_log"],
                x["anchor_index"],
            ),
        )
        for row in assets:
            if allowed.get(row["token"]) != row["source_log"]:
                raise ValueError("Asset outside navval")
        queue = []
        for row in failures:
            ready = q.db.execute(
                "SELECT status FROM vectors WHERE profile=? AND token=?", (profile, row["token"])
            ).fetchone()
            queue.append(
                dict(
                    token=row["token"],
                    pdms=row["metrics"]["PDMS"],
                    ready=bool(ready and ready[0] == "ready"),
                )
            )
        queries = [x["token"] for x in queue if x["ready"]][:max_queries]
        write_json(out / "private-failure-queue.json", dict(rows=queue, publishable=False))
        if not queries:
            raise ValueError("No ready vectors among observed failures")
        broad = retrieve(q, t, profile, queries, limit=8, progress=out / "retrieval-progress.json")
        write_json(out / "private-unrestricted-retrieval.json", broad)
        eligible = []
        missing = []
        for row in assets:
            try:
                v = ts.get(row["token"], profile)
            except ValueError:
                missing.append(dict(token=row["token"], reason="training_feature_missing"))
                continue
            values = normalize(
                np.frombuffer(v["image_vectors"], dtype="<f2").reshape(v["views"], v["dimension"])
            )
            eligible.append((row, values))
        ranks = []
        for query in queries:
            v = qs.get(query, profile)
            values = normalize(
                np.frombuffer(v["image_vectors"], dtype="<f2").reshape(v["views"], v["dimension"])
            )
            ranked = sorted(
                [
                    dict(asset=row, similarity=float((values * ev).sum(-1).mean()))
                    for row, ev in eligible
                    if ev.shape == values.shape
                ],
                key=lambda r: (-r["similarity"], r["asset"]["token"]),
            )
            ranks.append(dict(query_token=query, candidates=ranked))
        write_json(out / "private-asset-ranked.json", dict(rows=ranks, publishable=False))
        # One source per pass before considering another anchor of that source.
        selected = []
        seen = set()
        counts = {}
        for per_source in range(1, 5):
            for rank in range(len(eligible)):
                for row in ranks:
                    if rank >= len(row["candidates"]):
                        continue
                    item = row["candidates"][rank]
                    asset = item["asset"]
                    token = asset["token"]
                    group = asset["source_group"]
                    if token in seen or counts.get(group, 0) >= per_source:
                        continue
                    selected.append(asset)
                    seen.add(token)
                    counts[group] = counts.get(group, 0) + 1
                    if len(selected) >= max_candidates:
                        break
                if len(selected) >= max_candidates:
                    break
            if len(selected) >= max_candidates:
                break
        context = dict(
            model="augustus",
            parent_checkpoint=digest(Path(config["checkpoint"]) / "model.safetensors"),
            evaluator_version="native_navsim_v1_pdm_40x0.1_full_coverage",
            benchmark="navtest",
            selection_role="private_full_test_vector_retrieval_to_navval",
        )
        skill = assignment_plan(selected, context)
        write_json(out / "skill-plan.json", skill)
        result = dict(
            kind="navval_reactive_selection",
            inputs=frozen,
            assets=selected,
            plan=str((out / "skill-plan.json").resolve()),
            failure_count=len(queue),
            zero_count=sum(x["pdms"] == 0 for x in queue),
            queries=len(queries),
            selected=len(selected),
            missing_vectors=missing,
            claim="visual_similarity_not_confirmed_student_failure",
            publishable=False,
            training_pixels_from_test=False,
            production_route="driverl_new_history_gs",
        )
        write_json(out / "selection.json", result)
        return result
    finally:
        q.close()
        t.close()


def select_full_training(
    config_path,
    mining_root,
    coverage_path,
    output,
    *,
    max_candidates=16,
    preferred_bundles=(),
    exclude_tokens=(),
    generation_skill="driverl_full_window",
    require_localization_free=False,
):
    """Consume full-corpus hybrid matches; asset readiness does not alter retrieval."""
    from driversi.rsi.experiments.continuation import frozen_json
    from driversi.rsi.experiments.navval_retrieval import training_members
    from driversi.rsi.experiments.reactive_generation import assignment_plan

    if not 1 <= max_candidates <= 16:
        raise ValueError("Finite production wave required")
    if generation_skill not in ("driverl_full_window", "driverl_sample_full_window", "perturbed_history_tts"):
        raise ValueError("Unknown full-training generation skill")
    c = read_mapping(config_path)
    mining, out = Path(mining_root), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    summary = read_mapping(mining / "summary.json")
    provenance = read_mapping(mining / "inputs.json")
    if require_localization_free:
        from driversi.rsi.decisions.augmentation_budget import require_localization_free as validate_selection

        validate_selection(summary, provenance, read_mapping(mining / "training-production-queue.json"))
    elif summary.get("geography_used_for_selection") is True:
        raise ValueError("Geographic selection is audit-only, not a production source")
    full = Path(c["output"]) / "baseline"
    if (
        summary.get("status") != "completed"
        or summary.get("training_members") != 103288
        or provenance.get("population") != "official_navtrain"
        or provenance.get("config_sha256") != digest(config_path)
        or provenance.get("baseline_sha256") != digest(full / "report.json")
    ):
        raise ValueError("Verified full-training mapping for this parent is required")
    inputs = {
        str(p.resolve()): digest(p)
        for p in [
            Path(config_path),
            Path(coverage_path),
            mining / "inputs.json",
            mining / "summary.json",
            mining / "training-production-queue.json",
        ]
    }
    declaration = dict(
        files=inputs,
        max_candidates=max_candidates,
        preferred_bundles=list(preferred_bundles),
        exclude_tokens=sorted(exclude_tokens),
        **(
            {"generation_skill": generation_skill}
            if generation_skill != "driverl_full_window"
            else {}
        ),
        **({"localization_used_for_selection": False} if require_localization_free else {}),
    )
    frozen_json(out / "inputs.json", declaration)
    if (out / "selection.json").exists():
        value = read_mapping(out / "selection.json")
        if value["inputs"] != declaration or digest(value["plan"]) != value["plan_sha256"]:
            raise ValueError("Frozen full-training selection changed")
        return value
    catalog = Catalog(c["train_catalog"], "navtrain", readonly=True)
    try:
        allowed, _ = training_members(catalog, "navtrain")
    finally:
        catalog.close()
    coverage = read_mapping(coverage_path)
    if coverage.get("population") != "navtrain" or coverage.get("training_members") != len(allowed):
        raise ValueError("Full-training asset inventory required")
    for path, sha in coverage["files"].items():
        if digest(path) != sha:
            raise ValueError("Asset coverage input changed")
    assets = {}
    for row in sorted(
        coverage["rows"],
        key=lambda x: (x["bundle_root"] not in preferred_bundles, x["bundle_root"]),
    ):
        if allowed.get(row["token"]) != row["source_log"]:
            raise ValueError("Asset outside official training membership")
        assets.setdefault(row["token"], row)
    queue = read_mapping(mining / "training-production-queue.json")
    if queue.get("contains_test_tokens") is not False:
        raise ValueError("Training-only candidate queue required")
    ready, pending, seen = [], [], set()
    for rank, row in enumerate(queue["rows"], 1):
        if allowed.get(row["token"]) != row["source_log"] or row["token"] in seen:
            raise ValueError("Duplicate or invalid training candidate")
        seen.add(row["token"])
        if row["token"] in exclude_tokens:
            continue
        if row["token"] in assets:
            ready.append(dict(assets[row["token"]], retrieval_queue_rank=rank))
        else:
            pending.append(
                dict(
                    token=row["token"],
                    source_log=row["source_log"],
                    retrieval_queue_rank=rank,
                    status="needs_reconstruction_asset",
                    trainable=False,
                )
            )
    write_json(
        out / "pending-assets.json",
        dict(
            rows=pending,
            execution_supported=False,
            reason="No unattended reconstruction backend is registered",
            contains_test_tokens=False,
        ),
    )
    selected, used, counts = [], set(), {}
    for per_group in range(1, 5):
        for asset in ready:
            group = asset["source_group"]
            if asset["token"] in used or counts.get(group, 0) >= per_group:
                continue
            selected.append(asset)
            used.add(asset["token"])
            counts[group] = counts.get(group, 0) + 1
            if len(selected) >= max_candidates:
                break
        if len(selected) >= max_candidates:
            break
    context = dict(
        model=c.get("student_model", "augustus"),
        parent_checkpoint=(
            str(Path(c["checkpoint"]).resolve())
            if c.get("student_model") == "chainflow_stage1"
            else digest(Path(c["checkpoint"]) / "model.safetensors")
        ),
        evaluator_version="native_navsim_v1_pdm_40x0.1_full_coverage",
        benchmark="navtest",
        selection_role=(
            "private_full_test_scene_cluster_to_full_navtrain"
            if summary.get("context_channel")
            else "private_full_test_vector_retrieval_to_full_navtrain"
            if summary.get("text_channel", "").startswith("disabled")
            else "private_full_test_hybrid_retrieval_to_full_navtrain"
        ),
    )
    plan = assignment_plan(selected, context, skill=generation_skill)
    frozen_json(out / "skill-plan.json", plan)
    result = dict(
        kind="full_training_reactive_selection",
        inputs=declaration,
        assets=selected,
        plan=str((out / "skill-plan.json").resolve()),
        plan_sha256=digest(out / "skill-plan.json"),
        training_members=len(allowed),
        mapped_queries=summary["mapped_queries"],
        selected=len(selected),
        zero_count=summary["zero_count"],
        nonzero_below_06=summary["nonzero_below_06"],
        pending_assets=len(pending),
        excluded_prior_attempts=len(set(exclude_tokens) & seen),
        full_corpus_membership=True,
        full_corpus_vector_coverage=summary["ready_vectors"] == len(allowed),
        ready_vectors=summary["ready_vectors"],
        missing_training_vectors=len(allowed) - summary["ready_vectors"],
        asset_filtered_production=True,
        claim="similarity_is_not_confirmed_student_failure",
        publishable=False,
        training_pixels_from_test=False,
        production_route=generation_skill + "_gs",
    )
    write_json(out / "selection.json", result)
    return result
