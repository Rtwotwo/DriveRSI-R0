"""Small visual retrieval queues and a ten-sample-per-source Harness budget.

Global coordinates, map identity, geographic distance and absolute heading never
enter the production selector. Scores select test queries only. Production rows
contain training identities; test correspondences stay in private provenance.
"""

import argparse
import csv
import json
import math
import os
import socket
import time
from collections import Counter, defaultdict
from pathlib import Path

from driversi.configuration import read_mapping
from driversi.data.catalog.scene_hierarchy import digest, save_json
from driversi.data.catalog.store import identity
from driversi.rsi.decisions.augmentation_budget import build_budget, require_localization_free


def select_ranked(rows, sources, k):
    """Strict allowlist; location fields cannot affect filtering, scores or ties."""
    if type(k) is not int or not 1 <= k <= 20:
        raise ValueError('TopK must be an integer from 1 to 20')
    clean = []
    for rank, row in enumerate(rows, 1):
        token, score = row['token'], float(row['similarity'])
        if token not in sources or not math.isfinite(score) or not -1.001 <= score <= 1.001:
            raise ValueError('Invalid training candidate or similarity')
        clean.append(dict(token=token, similarity=score, original_rank=rank))
    if len({r['token'] for r in clean}) != len(clean):
        raise ValueError('Duplicate ranked training token')
    clean.sort(key=lambda r: (-r['similarity'], r['token']))
    result, groups = [], set()
    for row in clean:
        source = sources[row['token']]
        if source['history_continuous'] is not True or source['source_group'] in groups:
            continue
        result.append(row)
        groups.add(source['source_group'])
        if len(result) == k:
            break
    return result


def aggregate(by_query, sources):
    support = defaultdict(list)
    for query, rows in by_query.items():
        for row in rows:
            support[row['token']].append(row)
    result = []
    for token, rows in support.items():
        source = sources[token]
        result.append(dict(token=token, source_log=source['source_log'],
            source_group=source['source_group'], matching_queries=len(rows),
            best_rank=min(r['original_rank'] for r in rows),
            max_similarity=max(r['similarity'] for r in rows), trainable=False))
    return sorted(result, key=lambda r: (-r['matching_queries'], r['best_rank'],
                                        -r['max_similarity'], r['token']))


def run(project, output, *, top_k=3):
    if top_k not in (3, 5, 20):
        raise ValueError('Production presets support3,5 or20 independent sources per query')
    root, out = Path(project).resolve(), Path(output).resolve()
    if not out.is_relative_to(root / 'exp') or (out / 'summary.json').exists():
        raise ValueError('Use a new experiment output under exp; completed queries need no rerun')
    out.mkdir(parents=True, exist_ok=True)
    private = out / 'private'
    private.mkdir(exist_ok=True)
    os.chmod(private, 0o700)
    started = time.monotonic()
    base_path = root / 'exp/chainflow-restart-32000-v1/baseline-config.json'
    base = read_mapping(base_path)
    baseline_path = Path(base['output']) / 'baseline/report.json'
    baseline = read_mapping(baseline_path)
    if not baseline.get('complete') or baseline.get('completed') != 12146:
        raise ValueError('Complete native benchmark required')
    score_path = root / 'exp/navtrain-navtest-campaign-v1/baseline9405/per-scene.csv'
    with score_path.open() as stream:
        scores = list(csv.DictReader(stream))
    observed = {r['token']: float(r['metrics']['PDMS']) for r in baseline['rows']}
    if len(scores) != 12146 or len(observed) != 12146:
        raise ValueError('Benchmark membership incomplete')
    for row in scores:
        if row['valid'] != 'True' or abs(float(row['score']) - observed[row['token']]) > 1e-12:
            raise ValueError('Benchmark score tables differ')
    queries = sorted(t for t, score in observed.items() if score == 0)
    if len(queries) != 239:
        raise ValueError('Expected 239 zero queries for frozen 9405 baseline')
    members_path = root / 'lib/retrieval/navtrain/centroids-v1/members.json'
    sources = {r['token']: r for r in json.loads(members_path.read_text())}
    if len(sources) != 103288 or set(queries) & sources.keys():
        raise ValueError('Training membership or split isolation failed')
    visual_path = root / 'exp/chainflow-restart-32000-v1/mapping/private-vector-retrieval.json'
    visual = read_mapping(visual_path)
    if visual['profile'] != base['profile'] or visual['training_samples_from_test'] is not False:
        raise ValueError('Wrong feature provenance')
    ranked = {r['query_token']: r['candidates'] for r in visual['rows']}
    for q in queries:
        for row in ranked[q]:
            if row['input_id'] != sources[row['token']]['input_id']:
                raise ValueError('Training visual input identity changed')
    coverage_path = root / 'exp/augustus-reactive-auto-v1/full-training-asset-coverage-v1.json'
    coverage = read_mapping(coverage_path)
    assets = {r['token'] for r in coverage['rows'] if Path(r['bundle_root']).is_dir()}
    comparisons, selections = {}, {}
    for k in sorted({1, 2, 3, top_k}):
        selected = {q: select_ranked(ranked[q], sources, k) for q in queries}
        rows = aggregate(selected, sources)
        selections[f'visual_top{k}'] = selected
        comparisons[f'visual_top{k}'] = dict(unique_seeds=len(rows),
            query_count=sum(bool(v) for v in selected.values()),
            pairs=sum(map(len, selected.values())),
            source_logs=len({r['source_log'] for r in rows}),
            registered_asset_tokens=sum(r['token'] in assets for r in rows))
    # Optional comparison only: project the complete UNGATED cluster ranking.
    # Verify full leaf membership; never import its geographic filtered methods.
    cluster_path = root / 'exp/chainflow9405-compact-retrieval-v1/private/rankings.json'
    pools_path = root / 'lib/cluster_caption/clusters/navtrain/pools.json'
    cluster = read_mapping(cluster_path)
    pools = read_mapping(pools_path)['pools']
    cluster_queries = {r['token']: r for r in cluster['queries']}
    for q in queries:
        if {r['token'] for r in cluster['cluster_rankings'][q]} != set(pools[str(cluster_queries[q]['primary_leaf'])]):
            raise ValueError('Cluster comparison is geographically or otherwise truncated')
    for k in [2, 3]:
        selected = {q: select_ranked(cluster['cluster_rankings'][q], sources, k) for q in queries}
        rows = aggregate(selected, sources)
        comparisons[f'cluster_visual_top{k}'] = dict(unique_seeds=len(rows),
            query_count=sum(bool(v) for v in selected.values()), pairs=sum(map(len, selected.values())),
            source_logs=len({r['source_log'] for r in rows}),
            registered_asset_tokens=sum(r['token'] in assets for r in rows))
    production_rows = aggregate(selections[f'visual_top{top_k}'], sources)
    # Existing failures affect production readiness, never visual retrieval rank.
    previous = root / 'exp/chainflow-expansion24-v1/campaign/trajectory-family-summary.json'
    family = read_mapping(previous)
    old_plan = read_mapping(family['plan'])
    old_tokens = {r['id']: r['token'] for r in old_plan['assignments']}
    repair = {old_tokens[r['candidate']] for r in family['rows']
              if r['status'] in {'visual_preflight_rejected', 'visual_rejected'}}
    budget = build_budget(production_rows, assets, repair)
    queue = dict(rows=production_rows, contains_test_tokens=False, trainable=False,
                  localization_used_for_selection=False, method=f'global_visual_top{top_k}',
                 query_scope='239_zero_score_queries', score_role='test_query_selection_only')
    budget['queue_identity'] = identity(queue)
    inputs = dict(population='official_navtrain', config_sha256=digest(base_path),
                  baseline_sha256=digest(baseline_path), localization_used_for_selection=False,
                  files={str(p): digest(p) for p in [base_path, baseline_path, score_path,
                         members_path, visual_path, coverage_path, previous, Path(family['plan'])]})
    summary = dict(status='completed', kind='localization_free_visual_queue', training_members=103288,
        ready_vectors=103288, mapped_queries=239, zero_count=239, nonzero_below_06=0,
        text_channel=f'disabled_in_production_visual_top{top_k}', localization_used_for_selection=False,
        geography_used_for_selection=False, future_targets_used_for_selection=False,
        navtrain_student_scoring=False, adaptive_evaluation=True, comparison=comparisons,
        cluster_comparison_role='complete_ungated_ranking_allowlisted_to_token_and_cosine_only',
        production_seeds=len(production_rows), target_qualified_samples=budget['target_qualified_samples'],
        asset_states=budget['state_counts'], semantic_accuracy=None,
        elapsed_seconds=time.monotonic()-started, policy=f'top{top_k}_one_source_group_per_query',
        training_pixels_from_test=False, query_details_publishable=False)
    require_localization_free(summary, inputs, queue)
    save_json(out / 'inputs.json', inputs)
    save_json(out / 'training-production-queue.json', queue)
    save_json(out / 'global-budget.json', budget)
    save_json(private / 'visual-query-mapping.json', dict(rows=selections, publishable=False))
    save_json(out / 'summary.json', summary)
    template = read_mapping(root / 'exp/chainflow-restart-32000-v1/campaign-config.json')
    wave_root = out / 'waves/wave-0001'
    template.update(output=str(wave_root), selection_mode='full_training_visual_bounded',
        production_host=socket.gethostname(), evaluation_host=socket.gethostname(),
        mining_root=str(out), global_budget=str(out / 'global-budget.json'),
        global_budget_sha256=digest(out / 'global-budget.json'), max_candidates=2,
        max_samples=20, max_rendered_branches=40, max_wall_seconds=21600,
        minimum_training_samples=10, generation_skill='driverl_full_window',
        enable_training_after_wave=False)
    template.pop('reuse_completed_generations', None)
    template['trajectory_family'].update(target_per_anchor=10, attempts_per_anchor=32,
        anchor_budget_mode='reserve_all_targets', stop_after_consecutive_duplicates=3,
        stop_after_visual_failures=1, stop_after_execution_failures=2)
    from driversi.rsi.experiments.reactive_cohort import validate_budget
    validate_budget(template)
    save_json(out / 'wave-0001-config.json', template)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--top-k', type=int, choices=(3,5,20), default=3)
    a = parser.parse_args()
    print(json.dumps(run(a.project, a.output, top_k=a.top_k), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
