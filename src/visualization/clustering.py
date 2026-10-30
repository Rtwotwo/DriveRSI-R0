"""Allowlisted cluster browsing projection; test members never leave private storage."""

from pathlib import Path

from .catalog import read_json, resolve_receipt, safe_text


def failure_focus(config, project, cluster_source):
    """Only aggregate counts can enter the scene browser; no test rows."""
    if not config.get('failure_mapping'):
        return None, {}
    from driversi.data.catalog.scene_hierarchy import digest

    payload = read_json(resolve_receipt(project, config['failure_mapping']))
    summary = payload['summary']
    if (summary['status'] != 'complete' or summary['image_vectors_used'] is not False
            or digest(cluster_path(project, config['clusters'])) != summary['cluster_catalog_sha256']):
        raise ValueError('Failure focus does not match the current frozen cluster catalog')
    nodes = payload['nodes']
    if ([n['id'] for n in nodes] != [n['id'] for n in cluster_source['nodes']]
            or any(n['train_count'] != c['train_count'] or n['test_count'] != c['test_count']
                   for n, c in zip(nodes, cluster_source['nodes'], strict=True))):
        raise ValueError('Failure focus cluster coverage changed')
    fields = ('primary_zero', 'primary_low', 'primary_total', 'expanded_zero',
              'expanded_low', 'expanded_total')
    aggregates = {n['id']: {key: int(n[key]) for key in fields} for n in nodes}
    public = {key: summary[key] for key in (
        'baseline_pdms', 'zero_queries', 'nonzero_below_06', 'queries',
        'primary_leaf_clusters', 'expanded_leaf_clusters', 'primary_training_pool',
        'ranked_training_candidates', 'uncertain_primary_queries')}
    public['model'] = safe_text(summary['model'])
    public['query_seconds'] = float(summary['timing']['query_pipeline_median_seconds'])
    public['load_seconds'] = float(summary['timing']['load_and_validate_seconds'])
    public['total_seconds'] = float(summary['timing']['total_wall_seconds'])
    if config.get('overlap_summary'):
        overlap = read_json(resolve_receipt(project, config['overlap_summary']))
        if (overlap.get('status') != 'complete' or not overlap.get('same_checkpoint')
                or not overlap.get('same_12146_scores') or overlap['queries'] != summary['queries']
                or any(r['cluster_count'] != summary['ranked_training_candidates']
                       for r in overlap['comparisons'])):
            raise ValueError('Overlap does not match the selected failure mapping')
        public['overlap'] = [
            {**{key: int(row[key]) for key in ('comparison_count', 'shared', 'cluster_only', 'comparison_only')},
             'name': safe_text(row['name']), 'jaccard': float(row['jaccard']),
             'fraction_of_comparison': float(row['fraction_of_comparison']),
             'same_query_with_overlap': int(row['same_query']['queries_with_shared_candidate'])}
            for row in overlap['comparisons']]
    return public, aggregates


def cluster_path(project, relative):
    path = (Path(project) / relative).resolve()
    library = (Path(project) / "lib/cluster_caption/clusters").resolve()
    if path.is_relative_to(library) and path.suffix == ".json":
        return path
    return resolve_receipt(project, relative)


def overview(config, project):
    if not config:
        return None
    path = cluster_path(project, config["summary"])
    if not path.exists():
        return None
    source = read_json(path)
    if source.get("status") != "complete" or source.get("fit_split") != "navtrain":
        return None
    return {"schema_version": 1, "status": "complete", "data_file": "clusters.json",
            "members": {s: int(source["members"][s]) for s in ("navtrain", "navtest")},
            "leaf_clusters": int(source["leaf_clusters"]), "level_clusters": source["level_clusters"],
            "leaf_size": source["leaf_size"], "out_of_train_radius": source["out_of_train_radius"],
            "single_source_leaves": int(source["single_source_leaves"]),
            "elapsed_seconds": float(source["elapsed_seconds"]),
            "scope": "JSONL文本＋当前状态＋地图拓扑；未使用图像相似性向量。训练侧拟合、测试侧冻结映射。Caption为待事实复核草稿，不代表失败原因或检索Recall。"}


def public_clusters(config, project):
    source = read_json(cluster_path(project, config["clusters"]), max_bytes=64 * 1024 * 1024)
    meta = overview(config, project)
    if not meta:
        raise ValueError("Clustering is not complete")
    focus, node_focus = failure_focus(config, project, source)
    result = []
    for node in source["nodes"]:
        reps = []
        for rep in node["representatives"]:
            if rep.get("split") != "navtrain":
                raise ValueError("Test scene detail cannot be published")
            reps.append({"split": "navtrain", "token": safe_text(rep["token"]),
                         "description": safe_text(rep["description"]),
                         "speed_kmh": rep["speed_kmh"],
                         "navigation_command": safe_text(rep["navigation_command"]),
                         "tags": [safe_text(t) for t in rep["tags"]],
                         "map_labels": [safe_text(t) for t in rep.get("map_labels", [])],
                         "caption_status": "machine_draft"})
        result.append({**{key: node[key] for key in (
            "id", "parent", "depth", "children", "train_count", "test_count", "source_groups",
            "mean_distance", "radius_p99", "speed_mean_kmh", "speed_range_kmh", "xy")},
            "stop_reason": safe_text(node.get("stop_reason", "")), "label": safe_text(node["label"]),
            "tags": [{"text": safe_text(t["text"]), "count": int(t["count"])} for t in node["tags"]],
            "map_tags": [{"text": safe_text(t["text"]), "count": int(t["count"])} for t in node.get("map_tags", [])],
            "representatives": reps, "failure_focus": node_focus.get(node['id'])})
    if not result or result[0]["train_count"] != meta["members"]["navtrain"] or result[0]["test_count"] != meta["members"]["navtest"]:
        raise ValueError("Cluster root coverage mismatch")
    return {"schema_version": 1, "summary": meta, "cuts": source["cuts"], "nodes": result,
            "failure_mapping": focus,
            "publication_scope": "training_examples_and_aggregate_test_counts_only"}
