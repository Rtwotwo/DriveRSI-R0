import json

import pytest

from driversi.data.catalog.scene_hierarchy import digest
from driversi.visualization.clustering import failure_focus


def fixture(root):
    source = {'nodes': [{'id': 0, 'train_count': 10, 'test_count': 2}]}
    p = root / 'lib/cluster_caption/clusters/clusters.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(source))
    summary = dict(status='complete', image_vectors_used=False, cluster_catalog_sha256=digest(p),
                   model='ChainFlow', baseline_pdms=94.0394, zero_queries=1, nonzero_below_06=1,
                   queries=2, primary_leaf_clusters=1, expanded_leaf_clusters=1,
                   primary_training_pool=10, ranked_training_candidates=3, uncertain_primary_queries=0,
                   timing=dict(query_pipeline_median_seconds=.1, load_and_validate_seconds=1, total_wall_seconds=2))
    row = {**source['nodes'][0], **{f'{a}_{b}': 1 if b != 'total' else 2
           for a in ['primary', 'expanded'] for b in ['zero', 'low', 'total']},
           'query_token': 'PRIVATE_TEST_SENTINEL'}
    receipt = root / 'exp/run/cluster-focus.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(dict(summary=summary, nodes=[row], private_test='PRIVATE_TEST_SENTINEL')))
    return source, p, {'clusters': str(p.relative_to(root)), 'failure_mapping': str(receipt.relative_to(root))}


def test_projection_strictly_allowlists_aggregates(tmp_path):
    source, _, config = fixture(tmp_path)
    result = failure_focus(config, tmp_path, source)
    assert 'PRIVATE_TEST_SENTINEL' not in json.dumps(result)
    assert result[0]['queries'] == 2
    assert result[1][0]['primary_total'] == 2


def test_projection_rejects_different_cluster_catalog(tmp_path):
    source, path, config = fixture(tmp_path)
    path.write_text('{}')
    with pytest.raises(ValueError):
        failure_focus(config, tmp_path, source)
