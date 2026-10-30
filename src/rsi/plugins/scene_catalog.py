"""Freeze training-only catalog retrieval as a DriveHarness candidate artifact."""

import json
from pathlib import Path

from driversi.data.catalog.store import Catalog, identity
from driversi.rsi.contracts import _keys
from driversi.rsi.workflow.results import result


def prepare(config, base, registry):
    config = dict(config)
    if config.get("score_run") is not None or config.get("score_below") is not None:
        raise ValueError("navtrain student scoring is disabled for Harness retrieval")
    config.setdefault("score_run", None)
    config.setdefault("score_below", None)
    required = ("database", "structured_tags", "visual_tags", "profile", "text", "limit",
                "score_run", "score_below")
    optional=tuple(k for k in ("vector","history_recovery_profile") if k in config)
    _keys(config, (*required, *optional),
          "scene catalog retrieval")
    if "history_recovery_profile" in config and "vector" not in config:
        raise ValueError("History recovery requires an explicit vector query")
    path = (base / config["database"]).resolve(strict=True)
    for name in ("structured_tags", "visual_tags"):
        if not isinstance(config[name], list) or not all(isinstance(x, str) for x in config[name]):
            raise ValueError("Catalog tags must be strings")
    if type(config["limit"]) is not int or not 1 <= config["limit"] <= 64:
        raise ValueError("Bounded Harness retrieval requires 1..64 scenes")
    from driversi_agents.chainflow.benchmark import official_filter

    _, official = official_filter("navtrain")
    # Check direct official membership even for an operator-supplied catalog.
    allowed = set(official["tokens"])
    allowed_logs = set(official["log_names"])
    catalog = Catalog(path, "navtrain", readonly=True)
    try:
        catalog.db.execute("BEGIN")
        source = catalog.db.execute("SELECT value FROM metadata WHERE key='source'").fetchone()
        if not source or not json.loads(source[0]).get("complete"):
            raise ValueError("Harness requires an imported complete official catalog")
        vector_report = None
        if "vector" in config:
            from driversi.data.catalog.vectors import VectorStore

            v = config["vector"]
            _keys(v, ("query_token", "profile", "shortlist", "exclude_source", "diverse_sources"),
                  "vector query")
            query = catalog.scene(v["query_token"])["record"]
            if query["token"] not in allowed or query["source_log"] not in allowed_logs:
                raise ValueError("Query exemplar outside official navtrain")
            if "history_recovery_profile" in config:
                from driversi.data.catalog.history_recovery import enable_recovery_view
                enable_recovery_view(catalog,v["profile"],config["history_recovery_profile"])
            vector_report = VectorStore(catalog).search(
                "training", v["query_token"], v["profile"], limit=config["limit"],
                shortlist=v["shortlist"], exclude_source=v["exclude_source"],
                diverse_sources=v["diverse_sources"], filters={
                    "tags": config["structured_tags"], "profile": config["profile"],
                    "visual_tags": config["visual_tags"], "text": config["text"],
                    "score_run": config["score_run"], "score_below": config["score_below"],
                },
            )
            if "history_recovery_profile" in config:
                from driversi.data.catalog.history_recovery import retrieval_provenance
                retrieval_provenance(catalog,vector_report,v["profile"],config["history_recovery_profile"])
            selected = [catalog.scene(row["token"]) for row in vector_report["results"]]
        else:
            selected = catalog.query("training", config["structured_tags"], config["limit"],
                                     profile=config["profile"], visual_tags=config["visual_tags"],
                                     text=config["text"], score_run=config["score_run"],
                                     score_below=config["score_below"])
        candidates = []
        for row in selected:
            r = row["record"]
            if r["token"] not in allowed or r["source_log"] not in allowed_logs:
                raise ValueError("Training candidate outside official navtrain")
            annotations = []
            if config["profile"]:
                annotation = catalog.annotation(r["token"], config["profile"])
                annotations = [annotation] if annotation else []
            candidates.append({
                "token": r["token"], "source_log": r["source_log"], "record": r,
                "history_descriptor": row["descriptor"], "visual_observations": annotations,
                "model_cause_established": False, "trainable": False,
                "next_step": ("diagnose_condition_and_platform_validity"
                              if row["descriptor"].get("history",{}).get("continuous") is True
                              else "source_history_repair_required"),
            })
        report = {
            "schema_version": 1, "kind": "navtrain_catalog_candidates", "split": "navtrain",
            "source_catalog": str(path), "query": config, "candidates": candidates,
            "annotation_semantics": "unverified_visual_observations_not_ground_truth",
            "test_samples_consumed": 0, "synthesis_authorized": False, "trainable": False,
        }
        if vector_report is not None:
            report["vector_retrieval"] = vector_report
        if config["profile"]:
            row = catalog.db.execute("SELECT config FROM profiles WHERE id=?", (config["profile"],)).fetchone()
            if row is None:
                raise ValueError("Unknown annotation profile")
            report["annotation_profile"] = json.loads(row[0])
        return {"report": report, "snapshot_id": identity(report)}, 0
    finally:
        catalog.close()


def execute(config, inputs, output, registry):
    if identity(config["report"]) != config["snapshot_id"]:
        raise ValueError("Prepared catalog candidate snapshot changed")
    output = Path(output)
    output.mkdir(parents=True)
    return result(output, {**config["report"], "snapshot_id": config["snapshot_id"]})



def _training_catalog(config, base):
    from driversi_agents.chainflow.benchmark import official_filter
    catalog = Catalog((base / config["database"]).resolve(strict=True), "navtrain", readonly=True)
    try:
        catalog.db.execute("BEGIN")
        source = catalog.db.execute("SELECT value FROM metadata WHERE key='source'").fetchone()
        if not source or not json.loads(source[0]).get("complete"):
            raise ValueError("Complete training catalog required")
        _, official = official_filter("navtrain")
        tokens, logs = set(official["tokens"]), set(official["log_names"])
        pilot = config.get("pilot_tokens")
        if pilot is not None:
            if (not isinstance(pilot, list) or not 1 <= len(pilot) <= 128
                    or len(set(pilot)) != len(pilot) or not set(pilot) <= tokens):
                raise ValueError("Bounded exact official navtrain pilot tokens required")
            catalog.db.execute("CREATE TEMP TABLE selected_tokens (token TEXT PRIMARY KEY)")
            catalog.db.executemany("INSERT INTO selected_tokens VALUES (?)", [(t,) for t in pilot])
            # The main retrieval collection may already own the vectors view.
            # Every search joins these filtered scenes; retain its provenance.
            for table in ("scenes", "tags"):
                catalog.db.execute(f"CREATE TEMP VIEW {table} AS SELECT * FROM main.{table} WHERE token IN (SELECT token FROM selected_tokens)")
        if any(r[0] not in tokens or r[1] not in logs for r in catalog.db.execute("SELECT token,source_log FROM scenes")):
            raise ValueError("Catalog contains nontraining sources")
        return catalog
    except BaseException:
        catalog.close()
        raise


def _candidate(row, catalog, capability):
    scene = catalog.scene(row["token"])
    continuous = scene["descriptor"].get("history", {}).get("continuous") is True
    return {**row, "source_log": scene["record"]["source_log"],
        "history_descriptor": scene["descriptor"], "capability": capability,
        "status": "condition_match_requires_diagnosis" if continuous else "source_history_repair_required",
        "synthesis_requested": continuous, "teacher_pdms_minimum": 95,
        "admission_required": ["same_anchor_teacher_execution", "counterfactual_pdms_ge_95",
                               "physical_integrity", "visual_alignment"],
        "model_cause_established": False, "student_failure_required": False, "trainable": False}


def prepare_expansion(config, base, registry):
    _keys(config, ("database", "capability_feedback", "vector_profile", "caption_profile",
                   "capabilities", "per_capability", "pool", "exemplars",
                   *(k for k in ("pilot_tokens",) if k in config)), "capability scene expansion")
    from driversi.configuration import read_mapping
    from driversi.rsi.decisions.capability_feedback import validate_feedback
    from driversi.rsi.decisions.capability_skills import CONDITIONS
    from driversi.data.catalog.hybrid import HybridRetriever
    value = read_mapping((base / config["capability_feedback"]).resolve(strict=True))
    # Strict aggregate projection; scene rows or embeddings never enter the report.
    feedback = validate_feedback(value.get("capability_feedback", value))
    active = {r["capability"] for r in feedback["slices"] if r["failures"]}
    selected = config["capabilities"]
    if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
            or not set(selected) <= active):
        raise ValueError("Requested capabilities must have aggregate deficit evidence")
    if type(config["per_capability"]) is not int or not 1 <= config["per_capability"] <= 16:
        raise ValueError("Bounded per-capability quota required")
    catalog = _training_catalog(config, base)
    try:
        retriever = HybridRetriever(catalog, config["vector_profile"], config["caption_profile"])
        candidates, routes = [], {}
        for capability in selected:
            conditions, text = CONDITIONS[capability]
            report = retriever.search_conditions(conditions, text, limit=config["per_capability"],
                pool=config["pool"], exemplars=config["exemplars"])
            routes[capability] = report
            candidates.extend(_candidate(x, catalog, capability) for x in report["results"])
        output = {"kind": "navtrain_capability_candidates", "capability_feedback": feedback,
            "routes": routes, "candidates": candidates,
            "expansion_tokens": sorted({x["token"] for x in candidates if x["synthesis_requested"]}),
            "student_scores_required": False, "scores_used_for_ranking": False,
            "retrieval_scope": "bounded_official_training_pilot" if config.get("pilot_tokens") else "full_training_vector_population",
            "test_samples_consumed": 0, "evaluation_role": "adaptive_evaluation",
            "condition_mapping": "explicit_hypotheses_not_causal_findings",
            "synthesis_authorized": False, "trainable": False}
        return {"report": output, "snapshot_id": identity(output)}, 0
    finally:
        catalog.close()


def prepare_hybrid(config, base, registry):
    _keys(config, ("database", "query_token", "vector_profile", "caption_profile",
                   "limit", "pool", "target_category"), "score-free hybrid scene retrieval")
    from driversi.data.catalog.hybrid import HybridRetriever
    from driversi.rsi.decisions.capability_feedback import CAPABILITIES
    aliases = {"progress": "route_progress", "road": "road_compliance", "direction": "direction_compliance"}
    capability = aliases.get(config["target_category"], config["target_category"])
    if capability not in CAPABILITIES:
        raise ValueError("Known capability required")
    catalog = _training_catalog(config, base)
    try:
        report = HybridRetriever(catalog, config["vector_profile"], config["caption_profile"]).search(
            config["query_token"], limit=config["limit"], pool=config["pool"])
        candidates = [_candidate(row, catalog, capability) for row in report["results"]]
        output = {"kind": "navtrain_hybrid_candidates", "query": config, "retrieval": report,
            "candidates": candidates,
            "expansion_tokens": [x["token"] for x in candidates if x["synthesis_requested"]],
            "student_scores_required": False, "scores_used_for_ranking": False, "test_samples_consumed": 0,
            "synthesis_authorized": False, "trainable": False}
        return {"report": output, "snapshot_id": identity(output)}, 0
    finally:
        catalog.close()


def _reference(config, registry):
    from driversi.rsi.workflow.store import HarnessStore
    from driversi.rsi.workflow.artifacts import verify
    state = HarnessStore(registry).get(config["source_task"])
    stage = next(x for x in state["stages"] if x["id"] == config["source_stage"])
    declaration = next(x for x in state["task"]["steps"] if x["id"] == config["source_stage"])
    if stage["status"] != "completed" or declaration["output_kind"] != "navsim_scene_expansion_candidates":
        raise ValueError("Completed capability retrieval required")
    ref = verify(stage["result"], state["id"], declaration)
    return stage["result"], ref


def prepare_reference(config, base, registry):
    _keys(config, ("source_task", "source_stage"), "capability candidate reference")
    _, ref = _reference(config, registry)
    return {**config, "identity": ref["identity"]}, 0


def execute_reference(config, inputs, output, registry):
    original, ref = _reference(config, registry)
    if ref["identity"] != config["identity"]:
        raise ValueError("Referenced capability candidates changed")
    report = json.loads(Path(original["artifacts"]["report"]).read_text())
    output.mkdir(parents=True)
    return result(output, report)


def register(registry, Provider):
    registry.register(Provider("navsim_capability_candidates_reference", "1", (),
        "navsim_scene_expansion_candidates", prepare_reference, execute_reference, ("report",)))
    registry.register(Provider("navsim_scene_hybrid_retrieval", "2", (),
                               "navsim_scene_catalog_candidates", prepare_hybrid, execute, ("report",)))
    registry.register(Provider("navsim_scene_expansion_plan", "2", (),
                               "navsim_scene_expansion_candidates", prepare_expansion, execute, ("report",)))
    registry.register(Provider("navsim_scene_catalog_retrieval", "2", (),
                               "navsim_scene_catalog_candidates", prepare, execute, ("report",)))
