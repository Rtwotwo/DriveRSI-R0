"""Global augmentation targets, source readiness and finite native execution waves."""

from collections import Counter


def build_budget(rows, asset_tokens, repair_tokens, *, target=10, attempts=32, wave_size=2, global_target=None):
    global_mode=global_target is not None
    if global_mode and (global_target!=50 or target!=50):raise ValueError('Explicit global50 policy required')
    if type(target) is not int or not 10 <= target <= (50 if global_mode else 32):
        raise ValueError('Per-source qualified target must be 10..32')
    if type(attempts) is not int or not (1 if global_mode else target) <= attempts <= 64:
        raise ValueError('Attempt cap must cover target and not exceed 64')
    if type(wave_size) is not int or not 1 <= wave_size <= 10 or (not global_mode and wave_size * target > 128):
        raise ValueError('Bounded wave size required')
    tokens = [r['token'] for r in rows]
    if not rows or len(tokens) != len(set(tokens)):
        raise ValueError('Unique nonempty training-source queue required')
    states = []
    for row in rows:
        token = row['token']
        state = ('needs_asset_repair' if token in repair_tokens else
                 'asset_registered_unadmitted' if token in asset_tokens else 'needs_asset_build')
        states.append(dict(token=token, source_log=row['source_log'], state=state,
                           target_qualified=target, attempt_cap=attempts, trainable=False))
    counts = dict(Counter(r['state'] for r in states))
    count = len(rows)
    return dict(schema_version=1, kind='global_augmentation_budget', **({'global_target':global_target} if global_mode else {}),
                seeds=count, target_per_anchor=target, target_qualified_samples=global_target if global_mode else count * target,
                attempts_per_anchor=attempts, global_attempt_cap=count * attempts,
                render_cap_per_anchor=2 * target, global_render_cap=count * 2 * target,
                wave_size=wave_size, max_active_waves=1, states=states, state_counts=counts,
                stop_after_consecutive_duplicates=3, stop_after_visual_failures=1,
                stop_after_execution_failures=2, qualified_samples=0, attempts=0,
                semantic_precision=None, automatic_training=False,
                claim='production_targets_not_guaranteed_yield')


def wave_gate(budget, *, used_attempts=0, used_renders=0, unresolved=False):
    """The caller owns the global lease. Unknown outcomes never trigger new work."""
    if unresolved:
        return dict(allowed=False, reason='unresolved_prior_wave', selected_tokens=[])
    if any(type(n) is not int or n < 0 for n in (used_attempts, used_renders)):
        raise ValueError('Nonnegative measured counters required')
    ready = [r['token'] for r in budget['states'] if r['state'] == 'asset_registered_unadmitted']
    chosen = ready[:budget['wave_size']]
    if not chosen:
        return dict(allowed=False, reason='blocked_assets_or_repair', selected_tokens=[])
    if used_attempts + len(chosen) * budget['attempts_per_anchor'] > budget['global_attempt_cap']:
        return dict(allowed=False, reason='global_attempt_budget_exhausted', selected_tokens=[])
    if used_renders + len(chosen) * budget['render_cap_per_anchor'] > budget['global_render_cap']:
        return dict(allowed=False, reason='global_render_budget_exhausted', selected_tokens=[])
    return dict(allowed=True, reason='bounded_wave_available', selected_tokens=chosen,
                reserved_attempts=len(chosen) * budget['attempts_per_anchor'],
                reserved_renders=len(chosen) * budget['render_cap_per_anchor'])


def require_localization_free(summary, provenance, queue):
    """Reject coordinate-selected queues, including missing or ambiguous declarations."""
    for value in (summary, provenance, queue):
        if value.get('localization_used_for_selection') is not False:
            raise ValueError('Explicit localization-free selection evidence required')
        if value.get('geography_used_for_selection') is True:
            raise ValueError('Geography-selected candidates are audit-only')
    if queue.get('contains_test_tokens') is not False:
        raise ValueError('Training-only production queue required')


def runtime_gate(config):
    """Inspect real previous wave outcomes before allowing another bounded wave."""
    from pathlib import Path
    from driversi.configuration import read_mapping
    from driversi.agents.backends.native import digest
    from driversi.data.catalog.store import identity

    path = Path(config['global_budget'])
    if digest(path) != config['global_budget_sha256']:
        raise ValueError('Frozen global augmentation budget changed')
    budget = read_mapping(path)
    mining = Path(config['mining_root'])
    queue = read_mapping(mining / 'training-production-queue.json')
    require_localization_free(read_mapping(mining / 'summary.json'),
                              read_mapping(mining / 'inputs.json'), queue)
    if identity(queue) != budget['queue_identity']:
        raise ValueError('Global budget does not belong to this source queue')
    if config['max_candidates'] > budget['wave_size'] or (
        config['trajectory_family']['target_per_anchor'] != budget['target_per_anchor']
        or config['trajectory_family']['attempts_per_anchor'] != budget['attempts_per_anchor']
        or config['max_rendered_branches'] > budget['wave_size'] * budget['render_cap_per_anchor']
    ):
        raise ValueError('Wave exceeds global policy')
    used_attempts = used_renders = 0
    unresolved = False
    prior_tokens = set()
    current = Path(config['output']).resolve()
    for wave in sorted((path.parent / 'waves').glob('wave-*')):
        if wave.resolve() == current:
            continue  # Existing per-wave receipts govern continuation of this wave.
        status = wave / 'production-status.json'
        outcomes = wave / 'production-outcomes.json'
        if not status.exists():
            unresolved = True
            continue
        state = read_mapping(status)['status']
        if state not in {'completed', 'no_qualified_generated_samples', 'insufficient_qualified_samples',
                         'insufficient_per_anchor_samples', 'blocked_assets_or_repair',
                         'production_budget_blocked', 'generation_wave_completed',
                         'review_budget_exhausted', 'training_failed'}:
            unresolved = True
        if outcomes.exists():
            rows = read_mapping(outcomes)['rows']
            used_attempts += len(rows)
            used_renders += sum(bool(r.get('rendered')) for r in rows)
        selection = wave / 'selection' / 'selection.json'
        if selection.exists():
            prior_tokens.update(a['token'] for a in read_mapping(selection).get('assets', []))
    # A completed/failed source is not silently retried in a new wave.
    active = dict(budget, states=[r for r in budget['states'] if r['token'] not in prior_tokens])
    return dict(wave_gate(active, used_attempts=used_attempts, used_renders=used_renders,
                          unresolved=unresolved), used_attempts=used_attempts,
                used_renders=used_renders, state_counts=budget['state_counts'])
