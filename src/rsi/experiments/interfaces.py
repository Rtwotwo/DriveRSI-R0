"""Executable interface inventory and exact NAVSIM training-pool coverage.

An index is not a claim of asset readiness, training exposure or learning gain.
"""
import json
from pathlib import Path

from driversi.data.catalog.store import Catalog
from driversi.rsi.decisions.capability_skills import SKILLS
from driversi.rsi.decisions.vla_memory import SKILL_TREE
from driversi.rsi.workflow.providers import builtin_providers
from driversi.runtime.budget import write_json
from driversi.training.contracts.protocols import LIFECYCLE_BACKENDS


def interfaces():
    providers = {p['name']: p for p in builtin_providers().describe()}
    routes = {
        'chainflow_stage1': dict(mode='supervised', history=4, future=10,
            supervision=['actual_history_RGB', 'future_trajectory', 'same_branch_PDM_context'],
            synthetic_pair=True, blockers=['epoch_scale_training_not_validated',
                'native_candidate_promotion_and_automatic_recollection_not_connected']),
        'augustus': dict(mode='supervised', history=4, future=8,
            supervision=['actual_history_RGB', 'future_trajectory', 'branch_consistent_prediction_language'],
            synthetic_pair=False, blockers=['counterfactual_joint_cache_not_connected',
                'synthetic_paired_feedback_not_connected', 'GRPO_online_bridge_not_connected']),
        'driverl': dict(mode='online_rl', history=None, future=None,
            supervision=['current_policy_observation_action_logprob_value_reward_done_truncated'],
            synthetic_pair=False, blockers=['bounded_single_world_paper_derived_learner',
                'teacher_positive_trajectories_are_not_on_policy_samples']),
    }
    for model, route in routes.items():
        route['backend'] = LIFECYCLE_BACKENDS.get((model, route['mode']))
        route['full_scale_ready'] = False
    executable = []
    for name, recipe in SKILLS.items():
        provider = providers.get(recipe['provider'])
        executable.append(dict(id=name, recipe=recipe, provider_contract=provider,
            dispatch_connected=name in ('reactive_new_history', 'bounded_perturbation_history'),
            execution_entrypoint='scripts/rsi/run_capability_skill.py',
            memory_scope='experiment_evidence_not_driving_policy_weights',
            learned_driving_skill=False))
    return dict(kind='driversi_platform_interfaces', schema_version=2, models=routes,
        skills=executable, providers=list(providers.values()),
        vla_services=dict(tree=SKILL_TREE,provider=providers['vla_memory_skill_episode'],
            entrypoint='scripts/rsi/run_vla_memory_episode.py',
            memory_modes=['evolve','frozen'],learned_driving_skill=False),
        sequence=['official_training_pool', 'aggregate_capability_feedback',
                  'training_only_retrieval', 'versioned_skill_assignment',
                  'actual_new_history_and_teacher_future', 'trajectory_admission',
                  'same_branch_render', 'visual_admission', 'native_dataset_export',
                  'paired_training_resume_evaluation', 'evidence_memory_update'],
        time_contract=dict(physics_interval_seconds=.1, full_clip_seconds=7,
            arrival_seconds=2, exported_history_times=[-1.5, -1., -.5, 0.],
            chainflow_future_times=[.5*i for i in range(1, 11)],
            native_PDM_scored_seconds=4, executed_future_seconds=5,
            original_whitebox_intermediate_frames='offline_interpolation_not_sensor_observations'),
        memory_contract=dict(diagnostic='SceneSkillMemory: conditions, failures, evidence, applicable context',
            intervention='SkillMemory: versioned execution outcomes and verified paired gain',
            model_learning='only actual parameter updates and independent evaluation can establish learned behavior',
            frozen='retrieval_only; no writes or new credit',
            no_model_prompt_injection=True),
        training_caps=dict(
            legacy_pair=dict(per_rank_batch_size=1, paired_updates_max=128),
            epoch_pair=dict(implemented=True, explicit_opt_in='training.epoch_training',
                per_rank_batch_size_min=1, per_rank_batch_size_max=64, local_devices_max=8,
                update_budget='explicit even updates within common native scheduler horizon',
                sampling='global shuffled epoch with synthetic insertion every eight samples',
                resume='same world size, recipe, data identity and per-rank RNG required',
                full_103k_epoch_validated=False),
            synthetic_receipts_per_pair_max=64, automatic_promotion_validated=False),
        full_RSI_validated=False)


def validate_population(rows, official):
    """Exact membership and source agreement before writing any work index."""
    expected = set(official['tokens'])
    if len(expected) != len(official['tokens']):
        raise ValueError('Duplicate official training token')
    actual = {r['token'] for r in rows}
    if len(actual) != len(rows) or actual != expected:
        raise ValueError(f'Training catalog coverage mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')
    logs = set(official['log_names'])
    if any(r['source_log'] not in logs or not r['source_group'] for r in rows):
        raise ValueError('Training catalog source membership mismatch')


def write_pool(database, output, *, shards=8):
    from driversi_agents.chainflow.benchmark import official_filter
    if type(shards) is not int or not 1 <= shards <= 256:
        raise ValueError('Explicit shard count between1and256 required')
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new inventory output; existing work indexes are immutable')
    catalog = Catalog(database, 'navtrain', readonly=True)
    try:
        # Do not load images, scores, captions or large per-scene JSON payloads.
        rows = [dict(x) for x in catalog.db.execute(
            'SELECT token,source_log,source_group FROM scenes ORDER BY source_group,token')]
    finally:
        catalog.close()
    filter_path, official = official_filter('navtrain')
    validate_population(rows, official)
    groups = {g:i % shards for i,g in enumerate(sorted({x['source_group'] for x in rows}))}
    output.mkdir(parents=True)
    counts = [0]*shards
    with (output/'navtrain-work-index.jsonl').open('w') as stream:
        for row in rows:
            shard = groups[row['source_group']]; counts[shard] += 1
            stream.write(json.dumps(dict(**row, split='navtrain', shard=shard,
                catalog=str(Path(database).resolve()), status='indexed_not_executed'))+'\n')
    report = dict(kind='navtrain_pool_coverage', schema_version=1,
        catalog=str(Path(database).resolve()), official_filter=str(filter_path),
        expected_tokens=len(official['tokens']), indexed_tokens=len(rows),
        raw_source_groups=len(groups), missing_tokens=0, extra_tokens=0,
        shard_counts=counts, exact_membership=True, score_reads=0, image_reads=0,
        all_assets_checked=False, all_samples_trained=False,
        work_index=str(output/'navtrain-work-index.jsonl'))
    write_json(output/'coverage.json', report)
    write_json(output/'interfaces.json', interfaces())
    return report
