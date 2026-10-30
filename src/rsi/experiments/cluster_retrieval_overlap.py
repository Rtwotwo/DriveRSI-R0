"""Compare saved training candidate sets without rerunning either retriever."""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from driversi.data.catalog.scene_hierarchy import digest, save_json

from .cluster_failure_mapping import read_json, read_members


def set_overlap(a, b):
    a, b = set(a), set(b)
    shared = a & b
    return {'cluster_count': len(a), 'comparison_count': len(b),
            'shared': len(shared), 'cluster_only': len(a - b), 'comparison_only': len(b - a),
            'union': len(a | b), 'jaccard': len(shared) / len(a | b) if a | b else None,
            'fraction_of_comparison': len(shared) / len(b) if b else None,
            'fraction_of_cluster': len(shared) / len(a) if a else None}


def per_query_overlap(a, b, limit=None):
    if not a or set(a) != set(b):
        raise ValueError('Identical nonempty query sets required')
    rows = [set_overlap(a[q][:limit], b[q][:limit]) for q in sorted(a)]
    return {'queries': len(rows), 'queries_with_shared_candidate': sum(r['shared'] > 0 for r in rows),
            'mean_shared_candidates': float(np.mean([r['shared'] for r in rows])),
            'median_shared_candidates': float(np.median([r['shared'] for r in rows])),
            'mean_jaccard': float(np.mean([r['jaccard'] for r in rows])),
            'limit_per_query': limit, 'metric': 'agreement_not_relevance_recall'}


def query_dict(rows, key, candidates):
    result = {r[key]: [c['token'] if isinstance(c, dict) else c for c in r[candidates]] for r in rows}
    if len(result) != len(rows) or any(len(v) != len(set(v)) for v in result.values()):
        raise ValueError('Duplicate query or candidate token')
    return result


def run(config):
    started = time.perf_counter()
    run_root, old = Path(config['output']), Path(config['comparison_mapping'])
    out = run_root / 'overlap'
    out.mkdir(exist_ok=True)
    original = read_json(run_root / 'private/query-mapping.json')['rows']
    cluster = query_dict(original, 'token', 'training_candidates')
    visual_rows = read_json(old / 'private-vector-retrieval.json')['rows']
    hybrid_rows = read_json(old / 'private-hybrid-mapping.json')['rows']
    visual = query_dict(visual_rows, 'query_token', 'candidates')
    hybrid = query_dict(hybrid_rows, 'query_token', 'union')
    if set(cluster) != set(visual) or set(cluster) != set(hybrid):
        raise ValueError('Saved retrieval query populations differ')
    source = read_json(old / 'inputs.json')
    parent_config = read_json(source['config'])
    baseline = Path(parent_config['output']) / 'baseline'
    report = read_json(baseline / 'report.json')
    if (digest(baseline / 'report.json') != source['baseline_sha256']
            or digest(source['config']) != source['config_sha256']
            or digest(baseline / 'plan.json') != report['plan_sha256']
            or report['files'].get(parent_config['checkpoint']) != config['checkpoint_sha256']
            or not report.get('complete') or report.get('score_unit') != '0_to_1'
            or report.get('model') != 'chainflow_stage1'):
        raise ValueError('Old retrieval parent evidence differs')
    with Path(config['score_csv']).open() as f:
        score_rows = list(csv.DictReader(f))
    actual = {r['token']: float(r['score']) for r in score_rows}
    recorded = {r['token']: r['metrics']['PDMS'] for r in report['rows']}
    if (len(actual) != len(score_rows) or len(recorded) != len(report['rows'])
            or actual != recorded or report.get('completed') != len(actual)
            or report['files'].get(config['score_csv']) != digest(config['score_csv'])
            or set(cluster) != {t for t, score in actual.items() if score < .6}):
        raise ValueError('Old and new evaluation populations/scores differ')
    if {r['token']: r['score'] for r in original} != {t: actual[t] for t in cluster}:
        raise ValueError('Cluster mapping score provenance changed')
    train = read_members(Path(config['cluster_root']) / 'navtrain/members.jsonl', 'navtrain')
    train_by_token = {r['token']: r for r in train}
    nodes = read_json(Path(config['cluster_root']) / 'clusters.json')['nodes']
    group_by_query = {r['token']: r['group'] for r in original}
    a = set(t for tokens in cluster.values() for t in tokens)
    current_queue = {json.loads(line)['token'] for line in (run_root / 'training-candidates.jsonl').open()}
    old_queue = {r['token'] for r in read_json(old / 'training-production-queue.json')['rows']}
    if a != current_queue or set(t for tokens in hybrid.values() for t in tokens) != old_queue:
        raise ValueError('Saved queue is not the per-query candidate union')
    primary_leaves = {r['primary_leaf'] for r in original}
    expanded_leaves = {k for r in original for k in r['expanded_leaves']}
    primary_pool = {r['token'] for r in train if r['leaf'] in primary_leaves}
    expanded_pool = {r['token'] for r in train if r['leaf'] in expanded_leaves}
    comparisons = []
    for name, mapping in [('pure_visual_top16', visual), ('historical_hybrid_top16', hybrid)]:
        b = set(t for tokens in mapping.values() for t in tokens)
        if not (a | b) <= train_by_token.keys():
            raise ValueError('Non-training token in comparison')
        result = {'name': name, **set_overlap(a, b),
                  'same_query': per_query_overlap(cluster, mapping),
                  'comparison_lengths': dict(Counter(map(len, mapping.values()))),
                  'same_query_capped16': per_query_overlap(cluster, mapping, limit=16),
                  'same_query_by_group': {group: per_query_overlap(
                      {q: v for q, v in cluster.items() if group_by_query[q] == group},
                      {q: v for q, v in mapping.items() if group_by_query[q] == group})
                      for group in ['zero', 'low']},
                  'primary_pool_shared': len(b & primary_pool),
                  'expanded_pool_shared': len(b & expanded_pool),
                  'comparison_leaf_clusters': len({train_by_token[t]['leaf'] for t in b}),
                  'comparison_leaves_in_primary': len({train_by_token[t]['leaf'] for t in b} & primary_leaves)}
        shared_counts = Counter(train_by_token[t]['leaf'] for t in a & b)
        ac = Counter(train_by_token[t]['leaf'] for t in a)
        bc = Counter(train_by_token[t]['leaf'] for t in b)
        result['top_shared_clusters'] = [dict(leaf=k, label=nodes[k]['label'], shared=n,
               cluster_candidates=ac[k], comparison_candidates=bc[k])
               for k, n in sorted(shared_counts.items(), key=lambda x: (-x[1], x[0]))[:15]]
        for subset, tokens in [('shared', a & b), ('cluster-only', a - b), ('comparison-only', b - a)]:
            with (out / f'{name}-{subset}.csv').open('w') as f:
                writer = csv.writer(f)
                writer.writerow(['token', 'source_log', 'leaf'])
                for t in sorted(tokens):
                    writer.writerow([t, train_by_token[t]['source_log'], train_by_token[t]['leaf']])
        comparisons.append(result)
    result = {'status': 'complete', 'same_checkpoint': True, 'same_12146_scores': True,
              'same_527_queries': True, 'queries': len(cluster), 'requested_old_count': config.get('requested_comparison_count'),
              'verified_old_hybrid_count': len(old_queue), 'comparisons': comparisons,
              'interpretation': 'candidate_overlap_not_semantic_precision_recall_or_training_gain',
              'image_vector_retrieval_rerun': False, 'student_scoring_calls': 0,
              'elapsed_seconds': time.perf_counter() - started}
    save_json(out / 'summary.json', result)
    files = [old / n for n in ['inputs.json', 'private-vector-retrieval.json',
                               'private-hybrid-mapping.json', 'training-production-queue.json']]
    save_json(run_root / 'private/overlap-provenance.json',
              {'inputs': {str(p): digest(p) for p in files}, 'module_sha256': digest(__file__)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(read_json(args.config)), ensure_ascii=False))


if __name__ == '__main__':
    main()
