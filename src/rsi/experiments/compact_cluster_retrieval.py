"""Frozen text clusters followed by aligned visual ranking and explicit geo gates.

This does not refit clusters, score a student, or grant training admission. Test
identities and coordinates remain in private analysis; production receives only
training sources. Geography selected by this program is a selection condition,
not an independent semantic-accuracy measurement.
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from driversi.data.catalog.centroid_cache import open_cache
from driversi.data.catalog.scene_hierarchy import digest, save_json
from driversi.data.catalog.store import Catalog
from driversi.data.catalog.vectors import VectorStore, normalize
from driversi.rsi.experiments.retrieval_geography import pose_distance


def gate(pair, radius=25.0, heading=30.0):
    return (pair['same_map'] and pair['distance_m'] <= radius
            and pair['heading_difference_deg'] <= heading)


def choose(ranked, *, radius, continuous, uncertain=False, assets=None):
    if uncertain:
        return []
    return [r for r in ranked if gate(r, radius) and continuous.get(r['token']) is True
            and (assets is None or r['token'] in assets)][:1]


def compact_cover(rankings, sources, limit=24):
    """Greedy query coverage, one source log per seed, deterministic tie breaks.

Only supplied eligible pairs participate. Coverage denotes retrieval conditions,
not synthesis success. Never fill a budget with unrelated or already covered rows.
"""
    remaining = set(rankings)
    support = defaultdict(dict)
    for query, rows in rankings.items():
        for row in rows:
            support[row['token']][query] = row['similarity']
    selected, used_logs = [], set()
    while remaining and len(selected) < limit:
        choices = []
        for token, queries in support.items():
            covered = remaining.intersection(queries)
            if covered and sources[token]['source_log'] not in used_logs:
                choices.append((token, covered, float(np.mean([queries[q] for q in covered]))))
        if not choices:
            break
        token, covered, sim = min(choices, key=lambda x: (-len(x[1]), -x[2], x[0]))
        selected.append(dict(token=token, source_log=sources[token]['source_log'],
                             source_group=sources[token]['source_group'],
                             newly_covered_queries=len(covered), mean_similarity=sim,
                             trainable=False, admission_required=True))
        remaining -= covered
        used_logs.add(sources[token]['source_log'])
    return selected, len(rankings) - len(remaining)


def method_stats(by_query, poses, assets, radius=100.0):
    rows = [r for rs in by_query.values() for r in rs]
    good = [r for r in rows if gate(r, radius)]
    tokens = {r['token'] for r in rows}
    return dict(queries=len(by_query), queries_returned=sum(bool(r) for r in by_query.values()),
                pairs=len(rows), unique_training_tokens=len(tokens),
                source_logs=len({poses[t]['source_log'] for t in tokens}),
                registered_asset_tokens=len(tokens & assets),
                geographic_pass_fraction=len(good) / len(rows) if rows else None,
                geographic_hit_queries=sum(any(gate(r, radius) for r in rs) for rs in by_query.values()),
                geographic_protocol=dict(radius_m=radius, heading_deg=30),
                semantic_accuracy=None)


def run(project, output):
    root, out = Path(project).resolve(), Path(output).resolve()
    if not out.is_relative_to(root / 'exp'):
        raise ValueError('Experiment output must be under project exp')
    out.mkdir(parents=True, exist_ok=True)
    private = out / 'private'
    private.mkdir(exist_ok=True)
    os.chmod(private, 0o700)
    inputs = {
        'poses': root / 'exp/chainflow9405-retrieval-geography-v1/private/poses.json',
        'queries': root / 'exp/chainflow9405-cluster-mapping-v1/private/query-mapping.json',
        'pools': root / 'lib/cluster_caption/clusters/navtrain/pools.json',
        'visual': root / 'exp/chainflow-restart-32000-v1/mapping/private-vector-retrieval.json',
        'assets': root / 'exp/augustus-reactive-auto-v1/full-training-asset-coverage-v1.json',
    }
    provenance = {str(p): digest(p) for p in inputs.values()}
    if (out / 'summary.json').exists():
        old = json.loads((out / 'summary.json').read_text())
        if old['inputs'] != provenance or old['code_sha256'] != digest(__file__):
            raise ValueError('Frozen result changed; use a new output version')
        return old
    start = time.monotonic()
    read = lambda key: json.loads(inputs[key].read_text())
    poses = {r['token']: r for r in read('poses')['rows']}
    queries = [r for r in read('queries')['rows'] if r['score'] == 0]
    pools = read('pools')['pools']
    visual = {r['query_token']: r['candidates'] for r in read('visual')['rows']}
    coverage = read('assets')['rows']
    assets = {r['token'] for r in coverage if Path(r['bundle_root']).is_dir()}
    # Existing maps are immutable; actual GS quality remains a separate gate.
    train_manifest, members, _ = open_cache(root / 'lib/retrieval/navtrain/centroids-v1', verify=True)
    test_manifest, test_members, _ = open_cache(root / 'lib/retrieval/navtest/centroids-v1', verify=True)
    if train_manifest['profile'] != test_manifest['profile']:
        raise ValueError('Visual profiles differ')
    sources = {r['token']: r for r in members}
    if len(sources) != 103288 or len(test_members) != 12146 or len(queries) != 239:
        raise ValueError('Unexpected source/query coverage')
    query_tokens = {r['token'] for r in queries}
    if query_tokens & sources.keys():
        raise ValueError('Split overlap')
    needed = sorted({t for q in queries for t in pools[str(q['primary_leaf'])]})
    if set(needed) - sources.keys():
        raise ValueError('Cluster member missing from frozen training collection')
    continuous = {t: r['history_continuous'] for t, r in sources.items()}
    load_inputs_seconds = time.monotonic() - start
    save_json(out / 'progress.json', dict(phase='load_aligned_image_vectors', tokens=len(needed)))
    profile = train_manifest['profile']
    matrices, feature_ids = {}, {}
    for split, tokens in [('navtrain', needed), ('navtest', sorted(query_tokens))]:
        cat = Catalog(root / 'lib/retrieval' / split / 'catalog.sqlite', split, readonly=True)
        try:
            store = VectorStore(cat)
            for i, token in enumerate(tokens):
                row = store.get(token, profile)
                expected = sources[token]['input_id'] if split == 'navtrain' else next(
                    r['input_id'] for r in test_members if r['token'] == token)
                if row['input_id'] != expected:
                    raise ValueError('Frozen feature input changed')
                matrices[token] = normalize(np.frombuffer(row['image_vectors'], dtype='<f2').reshape(
                    row['views'], row['dimension']))
                feature_ids[token] = row['input_id']
                if i % 500 == 0:
                    save_json(out / 'progress.json', dict(phase='load_aligned_image_vectors',
                                                        split=split, completed=i, total=len(tokens)))
        finally:
            cat.close()
    load_seconds = time.monotonic() - start
    cluster_rankings = {}
    t = time.monotonic()
    for q in queries:
        token = q['token']
        ranked = []
        for candidate in pools[str(q['primary_leaf'])]:
            similarity = float((matrices[token] * matrices[candidate]).sum(axis=-1).mean())
            ranked.append(dict(token=candidate, similarity=similarity,
                               **pose_distance(poses[token], poses[candidate])))
        ranked.sort(key=lambda r: (-r['similarity'], r['token']))
        cluster_rankings[token] = ranked
    ranking_seconds = time.monotonic() - t
    methods = {k: {} for k in ['visual_top1', 'cluster_visual_top1', 'visual_geo25_top1',
                               'cluster_geo25_top1', 'cluster_geo50_top1', 'cluster_geo100_top1',
                               'cluster_geo25_asset_top1']}
    eligible, asset_eligible = {}, {}
    for q in queries:
        token = q['token']
        vr = [dict(token=r['token'], similarity=r['similarity'],
                   **pose_distance(poses[token], poses[r['token']])) for r in visual[token]]
        cr = cluster_rankings[token]
        methods['visual_top1'][token] = vr[:1]
        methods['cluster_visual_top1'][token] = cr[:1]
        methods['visual_geo25_top1'][token] = choose(vr, radius=25, continuous=continuous)
        for radius in [25, 50, 100]:
            methods[f'cluster_geo{radius}_top1'][token] = choose(cr, radius=radius,
                continuous=continuous, uncertain=q['out_of_train_radius'])
        eligible[token] = [r for r in cr if gate(r, 25) and continuous[r['token']] is True
                           and not q['out_of_train_radius']]
        asset_eligible[token] = [r for r in eligible[token] if r['token'] in assets]
        methods['cluster_geo25_asset_top1'][token] = asset_eligible[token][:1]
    selection_seconds = time.monotonic() - t - ranking_seconds
    compact, compact_coverage = compact_cover(eligible, sources)
    asset_compact, asset_compact_coverage = compact_cover(asset_eligible, sources)
    stats = {name: method_stats(rows, poses, assets) for name, rows in methods.items()}
    summary = dict(schema_version=1, status='complete', queries=239, query_role='zero_only',
        zero_query_leaves=len({q['primary_leaf'] for q in queries}),
        zero_query_outliers=sum(q['out_of_train_radius'] for q in queries),
        primary_pool_tokens=len(needed), registered_asset_pool=len(assets), methods=stats,
        selection=dict(radius_m=25, heading_deg=30, frozen_primary_leaf_only=True,
                       continuous_history_required=True, outlier_queries_excluded=True),
        compact=dict(seed_limit=24, seeds=len(compact), covered_queries=compact_coverage,
                     source_log_limit=1, asset_ready_seeds=sum(r['token'] in assets for r in compact)),
        asset_compact=dict(seeds=len(asset_compact), covered_queries=asset_compact_coverage),
        timings=dict(input_load_seconds=load_inputs_seconds, load_and_verify_seconds=load_seconds,
                     exact_aligned_ranking_seconds=ranking_seconds,
                     selection_seconds=selection_seconds, total_seconds=time.monotonic()-start),
        inputs=provenance, code_sha256=digest(__file__),
        semantic_accuracy=None, geography_used_for_selection=True, adaptive_evaluation=True,
        train_student_scoring=False, cluster_refitted=False, training_admitted=False,
        profile=profile, claim='fewer_geographically_related_seeds_not_verified_same_failure',
        failure_policy='no_related_asset_no_unrelated_fallback')
    save_json(private / 'rankings.json', dict(queries=queries, cluster_rankings=cluster_rankings,
              methods=methods, feature_inputs=feature_ids, publishable=False))
    save_json(out / 'compact-training-seeds.json', dict(rows=compact, trainable=False))
    save_json(out / 'asset-ready-training-seeds.json', dict(rows=asset_compact, trainable=False))
    save_json(out / 'summary.json', summary)
    save_json(out / 'progress.json', dict(phase='complete'))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.project, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
