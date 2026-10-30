"""Finite per-anchor native DriveRL candidate families before expensive GS calls."""

import copy
import time
import traceback
from collections import Counter
from pathlib import Path

from driversi.configuration import read_mapping
from driversi.data.catalog.store import identity
from driversi.rsi.decisions.trajectory_coverage import family_coverage
from driversi.rsi.decisions.trajectory_diversity import compare, describe
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


def coverage_diagnostics(descriptions):
    """An optional audit must never change execution/admission on diagnostic failure."""
    try:
        return family_coverage(descriptions)
    except Exception as exc:
        return dict(
            schema_version=1,
            kind="trajectory_family_coverage_diagnostics",
            status="diagnostic_failed",
            admission_effect="none",
            distinct_samples_certified=False,
            error_type=type(exc).__name__,
            reason=str(exc),
        )


class PendingGeneration(RuntimeError):
    """An owned physical/render task must settle before continuation; never restart it."""


def history_grid(policy):
    """Explicit, sealed references; grid size never certifies executed diversity."""
    from driversi.rsi.correction.history_perturbation import (
        perturbation_grid,
        validate_perturbation,
    )

    raw = policy.get("proposal_grid", perturbation_grid())
    minimum=1 if policy.get('cohort_mode')=='global_50_min1_v1' else 10
    if not isinstance(raw, list) or not minimum <= len(raw) <= 64:
        raise ValueError("Historical proposal grid is outside the declared finite cohort range")
    grid = [validate_perturbation(row) for row in raw]
    if len({tuple(row.values()) for row in grid}) != len(grid):
        raise ValueError("Duplicate historical endpoint references")
    return grid


def proposal_seeds(policy):
    count = policy["attempts_per_anchor"]
    seeds = policy.get("proposal_seeds")
    if seeds is None:
        return [policy["seed_start"] + i for i in range(count)]
    if (
        not isinstance(seeds, list)
        or len(seeds) != count
        or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Explicit proposal seeds must be unique and match candidate count")
    return list(seeds)


def family_training_gate(summary, anchor_ids, target, *, global_target=None):
    """Every selected source needs qualified rendered samples, not a global total."""
    if (
        type(target) is not int
        or target < 10
        or not anchor_ids
        or len(set(anchor_ids)) != len(anchor_ids)
    ):
        raise ValueError("Distinct selected anchors and per-anchor target >=10 required")
    if global_target is not None and global_target != 50:
        raise ValueError('Explicit global50 quota required')
    counts = {a: set() for a in anchor_ids}
    owners = {}
    reports = {}
    for row in summary["rows"]:
        aid = row["anchor_assignment_id"]
        if aid not in counts:
            raise ValueError("Unexpected source in family training evidence")
        if row.get("trainable"):
            if (
                row.get("status") != "training_sample_ready"
                or not row.get("rendered")
                or not row.get("sample_report")
            ):
                raise ValueError("Qualified family requires rendered, indexed native samples")
            package = row.get("dataset_reference", {}).get("package_id")
            if not package:
                raise ValueError("Qualified sample requires canonical dataset package identity")
            if global_target is not None and package in owners and owners[package] != aid:
                raise ValueError('Package is associated with multiple sources')
            if global_target is not None and row['sample_report'] in reports and reports[row['sample_report']]!=package:
                raise ValueError('Sample report represents multiple packages')
            owners[package] = aid
            reports[row['sample_report']] = package
            counts[aid].add(package)
    qualified = {a: len(v) for a, v in counts.items()}
    if sum(qualified.values()) != summary["qualified"]:
        raise ValueError("Family sample total differs from actual unique outcomes")
    minimum=1 if global_target is not None else target
    return dict(
        passed=all(n >= minimum for n in qualified.values()) and (global_target is None or sum(qualified.values()) >= global_target),
        target_per_anchor=target,
        qualified_per_anchor=qualified,
        shortages={a: max(0, minimum - n) for a, n in qualified.items()},
        **({'global_target':global_target,'global_shortage':max(0,global_target-sum(qualified.values()))} if global_target else {}),
        global_total_alone_is_not_sufficient=True,
    )


def render_budget_exhausted(config, outcomes, anchor_id, rendered):
    if rendered >= config['max_rendered_branches']:
        return True
    cap=config['trajectory_family'].get('render_cap_per_anchor')
    return cap is not None and sum(bool(row.get('rendered')) for row in outcomes
        if row['anchor_assignment_id']==anchor_id)>=cap


def coverage_slot_reserved(global_mode, qualified, assets, total, ready_count, anchor):
    return (global_mode and qualified[anchor]>0
        and total-ready_count<=sum(qualified[x]==0 for x in assets))


def validate_budget(c):
    policy = c["trajectory_family"]
    global_mode=policy.get('cohort_mode')=='global_50_min1_v1'
    if policy.get('cohort_mode') not in (None,'global_50_min1_v1'):
        raise ValueError('Unknown cohort mode')
    if global_mode and (policy.get('target_per_anchor')!=50 or c.get('max_samples')!=50
            or c.get('training_population')!='synthetic_only' or c.get('student_model')!='chainflow_stage1'):
        raise ValueError('Global mode requires an explicit 50-sample ChainFlow cohort')
    fields = (
        ("target_per_anchor", 10, 50 if global_mode else 32),
        ("attempts_per_anchor", 1 if global_mode else 10, 64),
        ("seed_start", 0, 2**31 - 65),
    )
    for key, low, high in fields:
        if type(policy[key]) is not int or not low <= policy[key] <= high:
            raise ValueError("Invalid trajectory-family " + key)
    proposal_seeds(policy)
    cap=policy.get('render_cap_per_anchor')
    if cap is not None and (type(cap) is not int or not policy['target_per_anchor']<=cap<=128):
        raise ValueError('Bounded per-source render allowance required')
    if not global_mode and policy["attempts_per_anchor"] < policy["target_per_anchor"]:
        raise ValueError("Attempt budget cannot be less than target")
    from driversi.rsi.correction.history_perturbation import FAMILY

    method = policy.get("method", "native_tts_seed_family_v1")
    if method not in (FAMILY, "native_tts_seed_family_v1", "native_policy_sample_family_v1"):
        raise ValueError("Unknown trajectory family method")
    if method == "native_policy_sample_family_v1" and c.get("generation_skill") != "driverl_sample_full_window":
        raise ValueError("Native sampling family requires its explicit skill")
    if method == FAMILY and (
        policy["attempts_per_anchor"] > len(history_grid(policy))
        or c.get("generation_skill") != "perturbed_history_tts"
    ):
        raise ValueError("History grid requires matching skill and bounded unique references")
    profile = c.get("visual_quality_profile", "experimental_numeric25_v1")
    if profile in (
        "experimental_numeric25_v1",
        "experimental_numeric24_ade03_v1",
        "experimental_numeric27_ade05_v1",
        "experimental_numeric27_strict_ade05_v1",
        "experimental_numeric25_spatial05_v1",
        "experimental_numeric25_adaptive_v1",
    ):
        from driversi.rendering.diagnostics.numeric_admission import numeric_settings

        minimum = numeric_settings(profile)["minimum_ade_m"]
    else:
        minimum = 0.5
    if policy.get("minimum_ade_m") != minimum:
        raise ValueError("Diversity threshold differs from the declared profile")
    extended = c.get("student_model") == "chainflow_stage1"
    if not 10 <= c["max_samples"] <= (128 if extended else 32) or not c["max_samples"] <= c[
        "max_rendered_branches"
    ] <= (512 if extended else 64):
        raise ValueError("Bounded multi-trajectory render/sample budget required")
    if not 1 <= c["max_candidates"] <= 16:
        raise ValueError("Bounded anchor budget required")
    mode = policy.get("anchor_budget_mode", "reserve_all_targets")
    if mode not in {"reserve_all_targets", "explore_until_global_limit"}:
        raise ValueError("Unknown anchor budget mode")
    if (
        mode == "reserve_all_targets"
        and c["max_candidates"] * policy["target_per_anchor"] > c["max_samples"]
        and not global_mode
    ):
        raise ValueError("Global sample quota cannot support per-anchor targets")
    for key in (
        "stop_after_consecutive_duplicates",
        "stop_after_visual_failures",
        "stop_after_execution_failures",
    ):
        value = policy.get(key)
        if value is not None and (
            type(value) is not int or not 1 <= value <= policy["attempts_per_anchor"]
        ):
            raise ValueError("Invalid bounded family stop policy: " + key)
    return policy


def family_stop_reason(rows, policy):
    """Budget routing only: no rejected sample becomes trainable or impossible."""
    if policy.get('stop_on_saturated_tail'):
        from driversi.rsi.decisions.scene_diversity_skill import saturated
        if saturated(rows):return 'source_recent_attempts_saturated'
    duplicate_limit = policy.get("stop_after_consecutive_duplicates")
    if (
        duplicate_limit
        and len(rows) >= duplicate_limit
        and all(
            r["status"] == "trajectory_duplicate_before_render" for r in rows[-duplicate_limit:]
        )
    ):
        return "consecutive_non_novel_native_rollouts"
    for key, states, reason in (
        (
            "stop_after_visual_failures",
            {"visual_rejected", "visual_preflight_rejected"},
            "visual_repair_required",
        ),
        (
            "stop_after_execution_failures",
            {"execution_failed", "visual_review_incomplete", "native_language_review_incomplete"},
            "execution_repair_required",
        ),
    ):
        limit = policy.get(key)
        if limit and sum(r["status"] in states for r in rows) >= limit:
            return reason
    return None


def expand_plan(original, policy, *, admission_profile=None, route_support_profile=None):
    if route_support_profile not in (None, "driverl_navigation_v2"):
        raise ValueError("Unknown route support profile")
    if admission_profile is not None:
        from driversi.rsi.correction.trajectory_admission import trajectory_policy

        trajectory_policy(admission_profile)
    from driversi.rsi.correction.history_perturbation import FAMILY

    method = policy.get("method", "native_tts_seed_family_v1")
    if method not in ("native_tts_seed_family_v1", "native_policy_sample_family_v1", FAMILY):
        raise ValueError("Unknown trajectory family method")
    grid = history_grid(policy) if method == FAMILY else None
    if grid and policy["attempts_per_anchor"] > len(grid):
        raise ValueError("History grid budget exceeds unique declared references")
    assignments = []
    seeds = proposal_seeds(policy)
    environments, grids, adaptive = {}, {}, {}
    if policy.get('environment_speed_context'):
        from driversi.rsi.decisions.environment_speed import VERSION, source_policy, adapt_grid
        context=policy['environment_speed_context']
        if not grid or context['version']!=VERSION:
            raise ValueError('Environment speed context requires the versioned history skill')
        for anchor in original['assignments']:
            env=source_policy(context['logs_root'],anchor['source_log'],anchor['token'])
            environments[anchor['id']]=env
            if policy.get('adaptive_context'):
                from driversi.rsi.decisions.generation_advisor import advise
                from driversi.rsi.decisions.adaptive_diversity import policy as compile_policy,proposal_grid
                advice_context=policy['adaptive_context'];advice=advise(advice_context,env)
                action=(advice_context['source_actions'].get(anchor['token'],'initial')
                        if 'source_actions' in advice_context else advice_context.get('action','initial'))
                preference='speed' if action=='adapt_spatial_or_speed_diversity' else advice['preference']
                if policy.get('scene_diversity'):
                    from driversi.rsi.decisions.scene_diversity_skill import source_context, compile_policy as scene_policy
                    ctx=source_context(policy['scene_diversity']['caption_database'],env,
                        advice_context.get('source_feedback',{}).get(anchor['token'],{}))
                    # Duplicate rejection alone does not imply that every source is speed-only.
                    settings=scene_policy(env,ctx,advice['preference'],advice_context.get('revision',0),action)
                else:
                    settings=compile_policy(env,preference,advice_context.get('revision',0),action)
                adaptive[anchor['id']]=settings
                grids[anchor['id']]=proposal_grid(env,settings,used=advice_context.get('used_proposals',{}).get(anchor['token'],[]))[:policy['attempts_per_anchor']]
            else:
                grids[anchor['id']]=adapt_grid(grid,env,policy['attempts_per_anchor'])
    for index in range(policy["attempts_per_anchor"]):
        for anchor in original["assignments"]:
            if grids and index>=len(grids[anchor['id']]):continue
            expected = "perturbed_history_tts" if grid else "driverl_sample_full_window" if method == "native_policy_sample_family_v1" else "driverl_full_window"
            if anchor["skill"] != expected:
                raise ValueError("Candidate family and executable skill differ")
            row = copy.deepcopy(anchor)
            row.update(
                id=f"{anchor['id']}-v{index:03d}",
                anchor_assignment_id=anchor["id"],
                sampling_seed=seeds[index],
                trajectory_family_version=method,
            )
            if grid:
                row["history_perturbation"] = grids[anchor['id']][index] if grids else grid[index]
            if environments:
                row['environment_speed_policy']=environments[anchor['id']]
            if adaptive:
                row['adaptive_diversity_policy']=adaptive[anchor['id']]
            if route_support_profile is not None:
                row["route_support_profile"] = route_support_profile
            if admission_profile is not None:
                row["admission_profile"] = admission_profile
            assignments.append(row)
    result = {k: copy.deepcopy(v) for k, v in original.items() if k != "plan_id"}
    result["assignments"] = assignments
    result["trajectory_family"] = dict(policy)
    if grids:
        result['proposal_capacity_per_anchor']={key:len(rows) for key,rows in grids.items()}
        result['proposal_capacity_shortages']={key:len(rows) for key,rows in grids.items()
            if len(rows)<(1 if policy.get('cohort_mode')=='global_50_min1_v1' else policy['target_per_anchor'])}
    if adaptive:
        from driversi.rsi.decisions.adaptive_diversity import PROPOSAL_VERSION
        result['adaptive_proposal_sampler']=PROPOSAL_VERSION
    return {**result, "plan_id": identity(result)}


def produce(selection, config, root, deadline, update_status, *, prefetch_workers=1):
    from driversi.rsi.experiments.reactive_prefetch import PhysicalPrefetch

    with PhysicalPrefetch(root, prefetch_workers) as prefetch:
        return _produce(selection, config, root, deadline, update_status, prefetch)


def _produce(selection, config, root, deadline, update_status, prefetch):
    from driversi.agents.backends.native import digest
    from driversi.rsi.experiments.reactive_generation import generate
    from driversi.rsi.experiments.reactive_postprocess import process

    c = read_mapping(config)
    if c.get("student_model") == "chainflow_stage1":
        from driversi.rsi.experiments.chainflow_campaign import process
    policy = validate_budget(c)
    global_mode=policy.get('cohort_mode')=='global_50_min1_v1'
    original = read_mapping(selection["plan"])
    plan = expand_plan(
        original,
        policy,
        admission_profile=c.get("trajectory_admission_profile"),
        route_support_profile=c.get("route_support_profile"),
    )
    if c.get("reuse_completed_generations"):
        references = {}
        for assignment in plan["assignments"]:
            key = assignment["token"] + ":" + str(assignment["sampling_seed"])
            if key in c["reuse_completed_generations"]:
                references[assignment["id"]] = c["reuse_completed_generations"][key]
        plan = {k: v for k, v in plan.items() if k != "plan_id"}
        plan["reused_executions"] = references
        plan["plan_id"] = identity(plan)
    plan_path = root / "trajectory-family-plan.json"
    frozen_json(plan_path, plan)
    assets = {
        a["id"]: asset
        for a, asset in zip(original["assignments"], selection["assets"], strict=True)
    }
    if plan.get('proposal_capacity_shortages'):
        summary=dict(status='completed_with_shortages',plan=str(plan_path),plan_sha256=digest(plan_path),
            attempted=0,qualified=0,rendered=0,rows=[],sample_reports=[],
            counts={'source_proposal_capacity_insufficient':len(plan['proposal_capacity_shortages'])},
            anchors=[dict(anchor=key,attempted=0,qualified=0,
                search_stop_reason='environment_feasible_proposals_below_target',
                proposal_capacity=plan['proposal_capacity_per_anchor'][key]) for key in assets],
            target_per_anchor=policy['target_per_anchor'],trainable=False)
        write_json(root/'trajectory-family-summary.json',summary)
        return []
    if c.get("resident_postprocessing"):
        from .resident_production import recover_completed
        recover_completed(root,plan)
    qualified, retained, outcomes = Counter(), {}, []
    from .problem_cohort import seed
    prior_rows=seed(c,plan)
    for row in prior_rows:
        aid=row['anchor_assignment_id']
        qualified[aid]+=1
        retained.setdefault(aid,[]).append(row['descriptor'])
    if c.get('prior_cohort'):
        # Samples withheld from training to reserve fresh slots remain novelty
        # references. Never pay to regenerate a discarded near-duplicate.
        pool=read_mapping(c['prior_cohort']['path'])  # seed() verified this reference.
        counted={r['sample_report'] for r in prior_rows}
        anchors={a['token']:a['id'] for a in original['assignments']}
        for row in pool['rows']:
            if row['source_token'] in anchors and row['sample_report'] not in counted:
                retained.setdefault(anchors[row['source_token']],[]).append(row['descriptor'])

    def execute(plan_path, cid, asset, database, directory, *, gpu, render=True):
        assignment = next(x for x in plan["assignments"] if x["id"] == cid)
        key = assignment["token"] + ":" + str(assignment["sampling_seed"])
        reference = c.get("reuse_completed_generations", {}).get(key)
        if reference is None:
            if not render and prefetch.has(cid):
                return prefetch.result(cid)
            return generate(
                plan_path,
                cid,
                asset,
                database,
                directory,
                gpu=gpu,
                render=render,
                **({"runtime": c["runtime"]} if c.get("runtime") else {}),
            )
        from driversi.rsi.plugins.synthesis import check_files

        path = Path(reference["path"]).resolve()
        if digest(path) != reference["sha256"]:
            raise ValueError("Cached generation receipt changed")
        saved = read_mapping(path)
        check_files(saved["files"])
        teacher = saved["teacher"]
        oldplan = read_mapping(teacher["plan_path"])
        if digest(teacher["plan_path"]) != teacher["plan_sha256"]:
            raise ValueError("Cached teacher plan changed")
        old = next(x for x in oldplan["assignments"] if x["id"] == teacher["assignment_id"])
        ignored = {"id", "anchor_assignment_id"}
        if {k: v for k, v in old.items() if k not in ignored} != {
            k: v for k, v in assignment.items() if k not in ignored
        }:
            raise ValueError("Cached physical controls or source differ from current assignment")
        job = read_mapping(path.parent / "generation.json")
        if saved["status"] != "rendered" or job["asset"]["bundle_root"] != asset["bundle_root"]:
            raise ValueError("Cached render asset or completion differs")
        directory.mkdir(parents=True, exist_ok=True)
        frozen_json(directory / "reuse.json", reference)
        frozen_json(directory / "report.json", saved)
        return dict(saved, status="rendered" if render else "trajectory_ready")

    ready, rendered = [r['sample_report'] for r in prior_rows], 0
    plan_order={a['id']:i for i,a in enumerate(plan['assignments'])}
    pending = []
    resident = c.get('resident_postprocessing')
    if resident and (c.get('student_model') != 'chainflow_stage1'
            or resident.get('batch_size') not in (2,16) or len(resident.get('gpus',[])) != 2):
        raise ValueError('Registered resident production requires two lanes and ChainFlow')
    trajectory_first = bool(resident and resident.get('collect_before_render'))
    if trajectory_first and (not global_mode or resident['collect_before_render']!=50):
        raise ValueError('Trajectory-first collection requires the global 50-sample cohort')
    def projected_counts():
        return qualified+Counter(a['anchor_assignment_id'] for a,_,_ in pending)
    def source_rows(aid):
        rows=[x for x in outcomes if x['anchor_assignment_id']==aid]
        rows += [dict(status='trajectory_pending_render',candidate=a['id']) for a,_,_ in pending if a['anchor_assignment_id']==aid]
        return sorted(rows,key=lambda x:x['candidate'])
    def queue_snapshot(stage='collecting_trajectories'):
        write_json(root/'trajectory-queue.json',dict(stage=stage,
            prepared=len(pending),admitted=len(ready),target=c['max_samples'],trainable=False,
            jobs=[dict(candidate=a['id'],source_token=a['token'],generation=str(d/'generation/report.json'),diversity=str(d/'diversity.json')) for a,d,_ in pending]))

    def snapshot(state):
        counts = Counter(x["status"] for x in outcomes)
        by_anchor = []
        for anchor in original["assignments"]:
            aid = anchor["id"]
            rows = [x for x in outcomes if x["anchor_assignment_id"] == aid]
            by_anchor.append(
                dict(
                    anchor=aid,
                    token=anchor["token"],
                    attempted=len(rows),
                    distinct_trajectories=len(retained.get(aid, [])),
                    coverage_diagnostics=coverage_diagnostics(
                        [x["descriptor"] for x in rows if x.get("descriptor")]
                    ),
                    qualified=qualified[aid],
                    target=1 if global_mode else policy["target_per_anchor"],
                    shortage=max(0, (1 if global_mode else policy["target_per_anchor"]) - qualified[aid]),
                    search_stop_reason=family_stop_reason(rows, policy),
                    search_stop_is_not_physical_impossibility=True,
                    behavior_types=dict(
                        Counter(t for x in rows for t in x.get("behavior_types", []))
                    ),
                )
            )
        value = dict(
            status=state,
            rows=prior_rows+outcomes,
            anchors=by_anchor,
            qualified=len(ready),
            prepared_trajectories=len(pending),
            collect_before_render=trajectory_first,
            reused_prior_cohort=len(prior_rows),
            rendered=rendered,
            attempted=len(outcomes),
            counts=dict(counts),
            target_per_anchor=policy["target_per_anchor"],
            anchor_budget_mode=policy.get("anchor_budget_mode", "reserve_all_targets"),
            sample_reports=ready,
            plan=str(plan_path),
            plan_sha256=digest(plan_path),
            target_met=bool(by_anchor) and all(x["shortage"] == 0 for x in by_anchor)
                and (not global_mode or len(ready)>=c['max_samples']),
            skills_certified=False,
            behavior_classification="kinematic_hypothesis_not_causal",
        )
        write_json(root / "production-outcomes.json", value)
        write_json(root / "trajectory-family-summary.json", value)
        from driversi.visualization.curriculum_operations import pulse
        pulse(root)
        return value

    def account(outcome, assignment, directory, *, already_retained=False):
        nonlocal rendered
        cid, aid = assignment['id'], assignment['anchor_assignment_id']
        terminal = directory/'outcome.json'
        consumed = (directory / "generation/render-reservation.json").exists() or (
            directory / "generation/reuse.json"
        ).exists()
        rendered += int(consumed)
        outcome["rendered"] = consumed
        if outcome.get("distinct") and not already_retained:
            retained.setdefault(aid, []).append(outcome["descriptor"])
        if outcome.get("trainable") is True:
            ready.append(outcome["sample_report"])
            qualified[aid] += 1
        outcomes.append(outcome)
        # Batch grouping changes completion order, never the training population order.
        outcomes.sort(key=lambda row:plan_order[row['candidate']])
        ready[:]=[r['sample_report'] for r in prior_rows]+[
            r['sample_report'] for r in outcomes if r.get('trainable') is True]
        from driversi.rsi.decisions.capability_skills import SkillMemory

        memory = SkillMemory(root / "generation-memory.sqlite")
        try:
            memory.record(
                dict(
                    evidence_id=digest(terminal),
                    outcome_type="data_generation",
                    context=plan["context"],
                    plan_id=plan["plan_id"],
                    capability=assignment.get("capability", "recovery"),
                    skill=assignment["skill"],
                    skill_version=assignment.get("skill_version", 1),
                    token=assignment["token"],
                    execution_status=outcome["status"],
                    behavior_types=outcome.get("behavior_types", []),
                    credited=False,
                    trainable=outcome.get("trainable", False),
                    reasons=["generation_is_not_student_learning_gain"],
                    report=str(terminal),
                )
            )
        finally:
            memory.close()
        snapshot("running")

    def flush_batch(batch):
        from .resident_production import dispatch
        queue_snapshot('batch_rendering_and_admission')
        update_status(root,'production','batch_rendering_and_admission',qualified_samples=len(ready),prepared_trajectories=len(pending))
        rows = dispatch(config, root, batch, deadline)
        indexed = {r['prepared_generation']:r['outcome'] for r in rows}
        for assignment, directory, outcome in batch:
            admitted=indexed[str(directory/'generation/report.json')]
            outcome.update(admitted, candidate=assignment['id'],
                anchor_assignment_id=assignment['anchor_assignment_id'],
                sampling_seed=assignment['sampling_seed'])
            if global_mode:
                if not (directory/'generation/render-reservation.json').exists():
                    raise ValueError('A global cohort render must retain its reservation')
                outcome['rendered']=True
            write_json(directory/'outcome.json', outcome)
            frozen_json(directory/'outcome-seal.json',dict(sha256=digest(directory/'outcome.json')))
            pending.remove((assignment,directory,outcome))
            account(outcome, assignment, directory, already_retained=True)
        queue_snapshot()

    def flush():
        while pending:
            aid=pending[0][0]['anchor_assignment_id']
            batch=[j for j in pending if j[0]['anchor_assignment_id']==aid][:resident['batch_size']]
            flush_batch(batch)

    for assignment_index, assignment in enumerate(plan["assignments"]):
        if time.time() >= deadline:
            snapshot("wall_budget_exhausted")
            raise TimeoutError("Production wall budget exhausted; no training dispatch")
        aid, cid = assignment["anchor_assignment_id"], assignment["id"]
        if pending and ((not trajectory_first and pending[0][0]['anchor_assignment_id'] != aid)
                or (trajectory_first and len(ready)+len(pending)>=c['max_samples'])):
            flush()
        directory = root / "candidates" / cid
        terminal = directory / "outcome.json"
        seal_path = directory / "outcome-seal.json"
        # Existing attempts must be replay-accounted even when a budget is now full.
        if not terminal.exists() and (
            qualified[aid] >= policy["target_per_anchor"] or len(ready) >= c["max_samples"]
        ):
            continue
        if (not terminal.exists() and coverage_slot_reserved(global_mode,projected_counts(),assets,
                c['max_samples'],len(ready)+len(pending),aid)):
            # Already-started physical work must settle, but cannot consume a
            # slot reserved for an uncovered source or be sent to rendering.
            if (directory/'pending.json').exists():
                raise PendingGeneration('Pending work requires source coverage reconciliation')
            if prefetch.has(cid):
                physical=prefetch.result(cid)
            elif (directory/'generation').exists():
                from .reactive_prefetch import completed_physical_result
                job=dict(plan_path=str(plan_path),assignment_id=cid,asset=assets[aid],
                    database=c['train_catalog'],output=str(directory/'generation'),
                    gpu=c.get('production_gpu',6),render=False,runtime=c.get('runtime'))
                physical=completed_physical_result(job,directory/'physical-prefetch.json')
                if physical is None:raise PendingGeneration('Prefetched source outcome not settled')
            else:
                continue
            report=directory/'generation/report.json'
            outcome=dict(candidate=cid,anchor_assignment_id=aid,sampling_seed=assignment['sampling_seed'],
                status='deferred_for_source_coverage',trainable=False,rendered=False,
                physical_status=physical['status'],files={str(report):digest(report)})
            write_json(terminal,outcome);frozen_json(seal_path,dict(sha256=digest(terminal)))
            account(outcome,assignment,directory)
            continue
        anchor_rows = source_rows(aid)
        if (
            not terminal.exists()
            and not (directory / "pending.json").exists()
            and family_stop_reason(anchor_rows, policy)
        ):
            # Keep prior receipts intact; the remaining seeds are unattempted,
            # not failed rollouts and not evidence the scenario is unsolvable.
            continue
        if (
            not terminal.exists()
            and not (directory / "pending.json").exists()
            and not (directory / "generation").exists()
            and c.get("review_budget_preflight") is True
        ):
            from driversi.rendering.diagnostics.gs_sequence_critic import review_capacity

            capacity = review_capacity(c["critic_config"], c["review_ledger"])
            write_json(root / "review-capacity.json", capacity)
            if not capacity["sufficient"]:
                snapshot("review_budget_exhausted")
                return ready
        directory.mkdir(parents=True, exist_ok=True)
        if not terminal.exists():
            jobs = []
            for future in plan["assignments"][assignment_index:]:
                faid, fcid = future["anchor_assignment_id"], future["id"]
                destination = root / "candidates" / fcid
                key = future["token"] + ":" + str(future["sampling_seed"])
                if (
                    (destination / "outcome.json").exists()
                    or key in c.get("reuse_completed_generations", {})
                    or qualified[faid] >= policy["target_per_anchor"]
                    or coverage_slot_reserved(global_mode,projected_counts(),assets,c['max_samples'],len(ready)+len(pending),faid)
                    or family_stop_reason(
                        source_rows(faid), policy
                    )
                ):
                    continue
                jobs.append(
                    (
                        fcid,
                        dict(
                            plan_path=str(plan_path),
                            assignment_id=fcid,
                            asset=assets[faid],
                            database=c["train_catalog"],
                            output=str(destination / "generation"),
                            gpu=c.get("production_gpu", 6),
                            render=False,
                            runtime=c.get("runtime"),
                        ),
                    )
                )
                if len(jobs) >= prefetch.workers:
                    break
            prefetch.schedule(jobs)
        update_status(
            root,
            "production",
            "generating_native_trajectory_family",
            candidate=cid,
            attempted=len(outcomes),
            qualified_samples=len(ready),
            prepared_trajectories=len(pending),
            rendered=rendered,
            anchor_qualified=qualified[aid],
            target_per_anchor=policy["target_per_anchor"],
        )
        if terminal.exists():
            if not seal_path.exists() or read_mapping(seal_path)["sha256"] != digest(terminal):
                raise ValueError("Unsettled or changed terminal candidate receipt")
            outcome = read_mapping(terminal)
            for path, sha in outcome.get("trajectory_files", {}).items():
                if digest(path) != sha:
                    raise ValueError("Persisted diversity evidence changed")
        else:
            outcome = dict(status="execution_failed", trainable=False)
            try:
                generated = execute(
                    plan_path,
                    cid,
                    assets[aid],
                    c["train_catalog"],
                    directory / "generation",
                    gpu=c.get("production_gpu", 6),
                    render=False,
                )
                outcome["status"] = generated["status"]
                if generated["status"] == "trajectory_ready":
                    description = describe(generated, cid)
                    novelty = compare(
                        description, retained.get(aid, []), minimum_ade_m=policy["minimum_ade_m"],
                        minimum_path_ade_m=(.5 if c.get("visual_quality_profile")=="experimental_numeric25_spatial05_v1" else None)
                    )
                    if c.get('visual_quality_profile')=='experimental_numeric25_adaptive_v1':
                        from driversi.rsi.decisions.adaptive_diversity import compare as adaptive_compare
                        novelty=adaptive_compare(description,retained.get(aid,[]),description['adaptive_diversity_policy'])
                    diversity_path = directory / "diversity.json"
                    if diversity_path.exists():
                        previous = read_mapping(diversity_path)
                        if "initial_heading_rad" not in previous["description"]:
                            # Resume old pending rendering without rewriting its sealed evidence.
                            # frozen_json still compares every original field and the decision.
                            description.pop("initial_heading_rad", None)
                    frozen_json(
                        diversity_path,
                        dict(
                            description=description,
                            decision=novelty,
                            **(
                                {"retained": retained.get(aid, [])}
                                if c.get("visual_quality_profile")
                                in (
                                    "experimental_numeric25_v1",
                                    "experimental_numeric24_ade03_v1",
                                    "experimental_numeric27_ade05_v1",
                "experimental_numeric27_strict_ade05_v1",
        "experimental_numeric25_spatial05_v1",
        "experimental_numeric25_adaptive_v1",
                                )
                                else {}
                            ),
                        ),
                    )
                    outcome.update(
                        behavior_types=description["behavior_types"],
                        trajectory_files=description["files"],
                        distinct=novelty["distinct"],
                        descriptor=description,
                        diversity=str(directory / "diversity.json"),
                    )
                    if not novelty["distinct"]:
                        outcome["status"] = "trajectory_duplicate_before_render"
                    elif render_budget_exhausted(c,
                        outcomes+[dict(anchor_assignment_id=a['anchor_assignment_id'],rendered=True)
                                  for a,_,_ in pending], aid, rendered+len(pending)):
                        outcome["status"] = "distinct_trajectory_pending_render_budget"
                    elif resident:
                        pending.append((assignment, directory, outcome))
                        retained.setdefault(aid, []).append(description)
                        queue_snapshot()
                        threshold=c['max_samples']-len(ready) if trajectory_first else min(resident['batch_size'],policy['target_per_anchor']-qualified[aid],c['max_samples']-len(ready))
                        if len(pending)>=threshold:flush()
                        continue
                    else:
                        generated = execute(
                            plan_path,
                            cid,
                            assets[aid],
                            c["train_catalog"],
                            directory / "generation",
                            gpu=c.get("production_gpu", 6),
                        )
                        if generated["status"] != "rendered":
                            outcome["status"] = generated["status"]
                        else:
                            admission = process(
                                directory / "generation/report.json",
                                config,
                                directory / "admission",
                            )
                            outcome.update(admission)
            except Exception as exc:
                if resident and (pending or seal_path.exists()
                        or (directory/'generation/render-reservation.json').exists()):
                    # A last-job accounting failure must never rewrite sealed batch success.
                    snapshot('resident_batch_requires_reconciliation')
                    raise
                (directory / "failure.txt").write_text(traceback.format_exc())
                outcome.update(
                    status="execution_failed", error_type=type(exc).__name__, trainable=False
                )
            outcome.update(
                candidate=cid, anchor_assignment_id=aid, sampling_seed=assignment["sampling_seed"]
            )
            if global_mode:
                outcome['rendered']=((directory/'generation/render-reservation.json').exists()
                    or (directory/'generation/reuse.json').exists())
            if outcome["status"] in {
                "teacher_outcome_unknown",
                "render_running",
                "render_registered",
            }:
                write_json(directory / "pending.json", outcome)
                outcomes.append(outcome)
                snapshot("awaiting_settled_generation")
                raise PendingGeneration("Reconcile the owned task without repeating execution")
            write_json(terminal, outcome)
            frozen_json(seal_path, dict(sha256=digest(terminal)))
        account(outcome, assignment, directory)
    flush()
    shortage=((len(ready)<c['max_samples'] or any(qualified[a]<1 for a in assets))
              if global_mode else any(qualified[a]<policy['target_per_anchor'] for a in assets))
    snapshot('completed_with_shortages' if shortage else 'target_met')
    return ready
