"""Map benchmark failures to frozen training caption/state/topology clusters."""

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from driversi.data.catalog.scene_hierarchy import (
    ancestor,
    assign_hierarchy,
    digest,
    nearest_leaves,
    save_json,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_members(path, split):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream]
    if any(r['split'] != split for r in rows) or len({r['token'] for r in rows}) != len(rows):
        raise ValueError('Wrong split or duplicate cluster members')
    return rows


def select_scores(rows, summary, test, checkpoint):
    """Require a complete, normalized, source-aligned parent evaluation."""
    expected = {r['token']: r['source_log'] for r in test}
    if (not summary.get('complete') or summary.get('evaluation_split') != 'navtest'
            or summary.get('metric') != 'NAVSIM_v1_PDMS'
            or summary.get('failed') != 0 or summary.get('duplicate_rows') != 0
            or summary.get('missing_tokens') or summary.get('unexpected_tokens')
            or summary.get('expected') != len(test) or summary.get('succeeded') != len(test)
            or summary.get('identity', {}).get('checkpoint_sha256') != checkpoint
            or len(rows) != len(test) or len({r['token'] for r in rows}) != len(rows)
            or {r['token'] for r in rows} != set(expected)):
        raise ValueError('Complete matching checkpoint evaluation required')
    selected, values = [], []
    for r in rows:
        score = float(r['score'])
        if (str(r['valid']).lower() != 'true' or r.get('error')
                or r['log_name'] != expected[r['token']]
                or not np.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('Invalid normalized score or source mismatch')
        values.append(score)
        if score < .6:
            selected.append({'token': r['token'], 'score': score,
                             'group': 'zero' if score == 0 else 'low'})
    if abs(float(np.mean(values)) * 100 - summary['pdms']) > 1e-7:
        raise ValueError('Score rows disagree with summary')
    return sorted(selected, key=lambda r: r['token'])


def measure(function, repeats=5):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return result, {'runs_seconds': samples, 'median_seconds': float(np.median(samples)),
                    'min_seconds': min(samples), 'max_seconds': max(samples)}


def pool_choices(primary, alternatives):
    return [list(dict.fromkeys([int(p), *map(int, alt)]))
            for p, alt in zip(primary, alternatives, strict=True)]


def ranked_candidates(query, pool, train_x, train, limit=20, per_source=2):
    pool = np.asarray(pool, dtype=np.int64)
    distances = np.linalg.norm(train_x[pool] - query, axis=1)
    order = np.lexsort((pool, distances))
    result, counts = [], Counter()
    for j in order:
        i = int(pool[j])
        log = train[i]['source_log']
        if counts[log] >= per_source:
            continue
        counts[log] += 1
        result.append((i, float(distances[j])))
        if len(result) == limit:
            break
    return result


def focus_counts(queries, primary, choices, nodes):
    counts = defaultdict(Counter)
    for q, leaf, expanded in zip(queries, primary, choices, strict=True):
        direct, expanded_ancestors = set(), set()
        for ids, leaves in [(direct, [int(leaf)]), (expanded_ancestors, expanded)]:
            for node_id in leaves:
                while node_id >= 0:
                    ids.add(node_id)
                    node_id = nodes[node_id]['parent']
        for kind, ids in [('primary', direct), ('expanded', expanded_ancestors)]:
            for node_id in ids:
                counts[node_id][kind + '_' + q['group']] += 1
                counts[node_id][kind + '_total'] += 1
    return counts


def run(config):
    from threadpoolctl import threadpool_limits

    from driversi_agents.chainflow.benchmark import official_filter

    started = time.perf_counter()
    root, out = Path(config['cluster_root']), Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    private = out / 'private'
    private.mkdir(exist_ok=True, mode=0o700)
    os.chmod(private, 0o700)
    files = [root / n for n in ['summary.json', 'hierarchy.json', 'hierarchy.npz',
             'clusters.json', 'navtrain/members.jsonl', 'navtest/members.jsonl',
             'navtrain/pools.json', 'navtrain-representation.npy', 'navtest-representation.npy']]
    files += [Path(config['score_csv']), Path(config['score_summary'])]
    source_digests = {str(p): digest(p) for p in files}
    meta, hierarchy, catalog = [read_json(root / n) for n in
                                ['summary.json', 'hierarchy.json', 'clusters.json']]
    if (meta.get('status') != 'complete' or meta.get('fit_split') != 'navtrain'
            or meta.get('image_vectors_used') is not False
            or hierarchy.get('fit_split') != 'navtrain'):
        raise ValueError('Frozen train-only non-image hierarchy required')
    train = read_members(root / 'navtrain/members.jsonl', 'navtrain')
    test = read_members(root / 'navtest/members.jsonl', 'navtest')
    if {r['token'] for r in train} & {r['token'] for r in test}:
        raise ValueError('Train/test overlap')
    for split, members in [('navtrain', train), ('navtest', test)]:
        _, spec = official_filter(split)
        if ({r['token'] for r in members} != set(spec['tokens'])
                or not {r['source_log'] for r in members} <= set(spec['log_names'])
                or len(members) != meta['members'][split]):
            raise ValueError('Cluster membership differs from official split')
    baseline = read_json(config['score_summary'])
    with Path(config['score_csv']).open() as stream:
        queries = select_scores(list(csv.DictReader(stream)), baseline, test,
                                config['checkpoint_sha256'])
    if not queries:
        raise ValueError('No zero/low queries in this evaluation')
    nodes = hierarchy['nodes']
    if [n['id'] for n in nodes] != list(range(len(nodes))):
        raise ValueError('Non-positional hierarchy IDs')
    leaves = [n['id'] for n in nodes if not n['children']]
    pools = defaultdict(list)
    for i, r in enumerate(train):
        if r['leaf'] not in leaves:
            raise ValueError('Training membership is not a leaf')
        pools[r['leaf']].append(i)
    expected_pools = {str(k): [train[i]['token'] for i in v] for k, v in pools.items()}
    if expected_pools != read_json(root / 'navtrain/pools.json')['pools']:
        raise ValueError('Training pool and members disagree')
    train_x, test_x = [np.load(root / f'{split}-representation.npy', allow_pickle=False)
                       for split in ['navtrain', 'navtest']]
    if (train_x.shape[0] != len(train) or test_x.shape[0] != len(test)
            or train_x.shape[1] != test_x.shape[1]
            or not np.isfinite(train_x).all() or not np.isfinite(test_x).all()):
        raise ValueError('Invalid stored representation')
    with np.load(root / 'hierarchy.npz', allow_pickle=False) as stored:
        centers = stored['centers'].copy()
        split_centers = dict(zip(map(int, stored['split_node_ids']), stored['split_centers'], strict=True))
    test_index = {r['token']: i for i, r in enumerate(test)}
    ids = [test_index[q['token']] for q in queries]
    qx = test_x[ids]
    load_seconds = time.perf_counter() - started
    with threadpool_limits(limits=config.get('threads', 8)):
        cached, lookup_time = measure(lambda: [test[test_index[q['token']]]['leaf'] for q in queries])
        (primary, _), route_time = measure(lambda: assign_hierarchy(qx, nodes, split_centers))
        (alternatives, _), nearest_time = measure(lambda: nearest_leaves(qx, centers[leaves], leaves, k=3))
        if list(map(int, primary)) != cached:
            raise ValueError('Recomputed primary mapping differs from saved assignments')
        cached_alts = [[v['leaf'] for v in test[i]['nearest_training_leaves']] for i in ids]
        if alternatives.tolist() != cached_alts:
            raise ValueError('Recomputed nearest centers differ from saved alternatives')
        choices = pool_choices(primary, alternatives)
        def expand():
            return [sorted(i for leaf in group for i in pools[leaf]) for group in choices]
        candidate_pools, expand_time = measure(expand)
        ranked, rank_time = measure(lambda: [ranked_candidates(q, p, train_x, train)
                                             for q, p in zip(qx, candidate_pools, strict=True)], repeats=3)
        def query_pipeline():
            route, _ = assign_hierarchy(qx, nodes, split_centers)
            adjacent, _ = nearest_leaves(qx, centers[leaves], leaves, k=3)
            groups = pool_choices(route, adjacent)
            return [ranked_candidates(q, sorted(i for leaf in group for i in pools[leaf]), train_x, train)
                    for q, group in zip(qx, groups, strict=True)]
        replay_ranked, pipeline_time = measure(query_pipeline, repeats=3)
        if ranked != replay_ranked:
            raise ValueError('Complete query pipeline replay differs')
    processing_start = time.perf_counter()
    counts = focus_counts(queries, primary, choices, nodes)
    cluster_rows = []
    global_rate = len(queries) / len(test)
    for n in catalog['nodes']:
        c = counts[n['id']]
        cluster_rows.append({
            'id': n['id'], 'label': n['label'], 'depth': n['depth'], 'leaf': not n['children'],
            'train_count': n['train_count'], 'test_count': n['test_count'],
            'source_groups': n['source_groups'], 'map_tags': n.get('map_tags', []),
            'speed_mean_kmh': n['speed_mean_kmh'],
            **{f'{kind}_{group}': c[f'{kind}_{group}'] for kind in ['primary', 'expanded']
               for group in ['zero', 'low', 'total']},
            'failure_rate': c['primary_total'] / n['test_count'] if n['test_count'] else None,
            'failure_enrichment': c['primary_total'] / n['test_count'] / global_rate if n['test_count'] else None,
        })
    leaf_focus = [r for r in cluster_rows if r['leaf'] and r['primary_total']]
    top = sorted(leaf_focus, key=lambda r: (-r['primary_total'], -r['primary_zero'], r['id']))
    level_counts = {}
    for level, depth in {**hierarchy['config']['cuts'], 'leaf': 999}.items():
        level_counts[level] = {}
        for group in ['zero', 'low', 'total']:
            ix = [i for i, q in enumerate(queries) if group == 'total' or q['group'] == group]
            level_counts[level][group] = len({ancestor(int(primary[i]), nodes, depth) for i in ix})
    primary_pool = set(i for leaf in primary for i in pools[int(leaf)])
    expanded_pool = set(i for p in candidate_pools for i in p)
    candidates = {}
    private_rows = []
    for q, leaf, group, ranking in zip(queries, primary, choices, ranked, strict=True):
        ti = test_index[q['token']]
        private_rows.append({**q, 'primary_leaf': int(leaf), 'expanded_leaves': group,
                             'out_of_train_radius': test[ti]['out_of_train_radius'],
                             'training_candidates': [train[i]['token'] for i, _ in ranking]})
        for rank, (i, distance) in enumerate(ranking, 1):
            t = train[i]
            c = candidates.setdefault(t['token'], {
                'token': t['token'], 'split': 'navtrain', 'source_log': t['source_log'],
                'leaf': t['leaf'], 'history_continuous': t['history_continuous'],
                'map_labels': t['map_labels'], 'query_hits': 0, 'zero_hits': 0, 'low_hits': 0,
                'best_rank': rank, 'min_representation_distance': distance,
                'student_failure_confirmed': False, 'training_admission_granted': False,
                'next_action': 'source_and_asset_preflight'})
            c['query_hits'] += 1
            c[q['group'] + '_hits'] += 1
            c['best_rank'] = min(c['best_rank'], rank)
            c['min_representation_distance'] = min(c['min_representation_distance'], distance)
    # Same-rank round robin prevents the first query taking the whole shortlist.
    fair, seen = [], set()
    for rank in range(max(map(len, ranked))):
        for ranking in ranked:
            if rank < len(ranking):
                token = train[ranking[rank][0]]['token']
                if token not in seen:
                    seen.add(token)
                    fair.append(candidates[token])
    with (out / 'training-candidates.jsonl').open('w') as stream:
        for row in fair:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    with (out / 'primary-training-pool.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['token', 'source_log', 'leaf'])
        for i in sorted(primary_pool):
            writer.writerow([train[i][k] for k in ['token', 'source_log', 'leaf']])
    save_json(private / 'query-mapping.json', {'publishable': False, 'rows': private_rows})
    os.chmod(private / 'query-mapping.json', 0o600)
    for p, expected in source_digests.items():
        if digest(p) != expected:
            raise ValueError('Inputs changed during mapping')
    summary = {
        'status': 'complete', 'model': 'ChainFlow stage1 navtrainval 9405',
        'baseline_pdms': baseline['pdms'], 'official_reference_pdms': 94.05,
        'cluster_catalog_sha256': source_digests[str(root / 'clusters.json')],
        'evaluation_members': len(test), 'training_members': len(train),
        'zero_queries': sum(q['group'] == 'zero' for q in queries),
        'nonzero_below_06': sum(q['group'] == 'low' for q in queries), 'queries': len(queries),
        'mapped_queries': len(private_rows), 'leaf_clusters_total': len(leaves),
        'primary_leaf_clusters': len(leaf_focus), 'expanded_leaf_clusters': len(set(v for c in choices for v in c)),
        'level_clusters': level_counts, 'primary_training_pool': len(primary_pool),
        'expanded_training_pool': len(expanded_pool), 'ranked_training_candidates': len(fair),
        'ranked_source_logs': len({r['source_log'] for r in fair}),
        'uncertain_primary_queries': sum(test[i]['out_of_train_radius'] for i in ids),
        'top_clusters': top[:15], 'source_files_unchanged': True,
        'replayed_primary_equal': True, 'replayed_nearest_equal': True,
        'adaptive_evaluation': True, 'image_vectors_used': False,
        'student_scoring_calls': 0, 'api_calls': 0, 'training_admission_granted': False,
        'semantic_recall': None, 'caption_quality': 'machine_drafts_not_fact_certified',
        'matching_meaning': 'similar_scene_conditions_not_confirmed_same_failure_cause',
        'timing': {
            'load_and_validate_seconds': load_seconds, 'stored_assignment_lookup': lookup_time,
            'hierarchy_route': route_time, 'three_nearest_leaf_search': nearest_time,
            'expand_training_pools': expand_time, 'within_pool_rank': rank_time,
            'query_pipeline': pipeline_time,
            'query_pipeline_median_seconds': pipeline_time['median_seconds'],
            'aggregate_export_and_recheck_seconds': time.perf_counter() - processing_start,
            'total_wall_seconds': time.perf_counter() - started,
            'scope': 'fresh_process_existing_frozen_representation_OS_cache_uncontrolled; warm stages repeated; no caption/fitting cost',
            'threads': config.get('threads', 8)},
    }
    save_json(out / 'summary.json', summary)
    save_json(out / 'cluster-focus.json', {'schema_version': 1, 'summary': summary, 'nodes': cluster_rows})
    save_json(private / 'provenance.json', {'inputs': source_digests, 'config': config,
              'module_sha256': digest(__file__), 'checkpoint_sha256': config['checkpoint_sha256']})
    save_json(out / 'complete.json', {'status': 'complete', 'summary': 'summary.json'})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(read_json(args.config)), ensure_ascii=False))


if __name__ == '__main__':
    main()
