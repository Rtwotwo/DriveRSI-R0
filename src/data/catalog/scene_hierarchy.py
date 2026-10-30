"""Exploratory scene hierarchy; source captions and admission flags stay read-only."""

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

FACTS = ("anchor_speed_kmh", "anchor_ax_mps2", "anchor_ay_mps2",
         "inside_intersection", "intersection_distance_m", "nearby_lanes_30m",
         "nearby_connectors_30m", "crosswalk_distance_m", "inside_lane_connector",
         "lane_incoming_edges", "lane_outgoing_edges", "lane_group_lanes")


def anchor_facts(caption, original):
    state = dict(re.findall(r"\b(vx|vy|ax|ay):\s*(-?[0-9]+(?:\.[0-9]+)?)", original.get("ego_dynamic_state", "")))
    command = str(caption.get("navigation_command", "")).upper()
    slot = {"GO STRAIGHT": 0, "GO FORWARD": 0, "TURN LEFT": 1, "TURN RIGHT": 2}.get(command, 3)
    return {"anchor_speed_kmh": caption.get("anchor_speed_kmh", np.nan),
            "anchor_ax_mps2": float(state["ax"]) if "ax" in state else np.nan,
            "anchor_ay_mps2": float(state["ay"]) if "ay" in state else np.nan,
            "nav_command_slot": slot}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_corpus(root, split, expected, topology_path=None):
    """Use finalized source JSONL, never the visual similarity database."""
    root = Path(root)
    library = root / f"lib/cluster_caption/{split}"
    original_path = library / ("autodrive-p3.original.json" if split == "navtrain" else "navsim.original.json")
    originals = json.loads(original_path.read_text())
    source_paths = sorted((library / "shards").glob("*.jsonl"))
    topology = {}
    if topology_path:
        with Path(topology_path).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["token"] in topology:
                    raise ValueError("Duplicate topology identity")
                topology[row["token"]] = row
    by_token, original_keys = {}, set()
    for shard in source_paths:
        for line_number, line in enumerate(shard.open(encoding="utf-8"), 1):
            row = json.loads(line)
            caption = row["cluster_caption"]
            token = caption["token"]
            if token in by_token or not caption or caption.get("token") != token:
                raise ValueError("Missing or duplicate caption identity")
            if caption.get("status") != "machine_draft" or caption.get("split") != split:
                raise ValueError("Caption status/split mismatch")
            if caption.get("observation_scope") != "current_anchor_four_cameras" or caption.get("future_images_included") is not False:
                raise ValueError("Unexpected caption information scope")
            key = row["annotation_key"]
            if key != row["scene_name"] + "_" + token or key not in originals:
                raise ValueError("Caption missing from original annotations")
            if any(row.get(k) != v for k, v in originals[key].items()):
                raise ValueError("Native annotation fields changed in source JSONL")
            original_keys.add(key)
            facts = anchor_facts(caption, row)
            topo = topology.get(token, {})
            facts.update(topo.get("features", {}))
            member = {"token": token, "source_log": row["scene_name"],
                      "source_group": re.sub(r"_[0-9]+_[0-9]+$", "", row["scene_name"]),
                      "features": facts, "map_labels": topo.get("labels", []),
                      "lane_association": topo.get("lane_association", "not_computed"),
                      "source_jsonl": str(shard.relative_to(root)), "source_line": line_number,
                      "information_set": "current_anchor_only", "history_continuous": None}
            by_token[token] = (caption, member)
    tokens = sorted(by_token)
    if len(tokens) != expected or len(originals) != expected or original_keys != set(originals):
        raise ValueError("Original/caption membership mismatch")
    if topology_path and set(tokens) != set(topology):
        raise ValueError("Topology/caption membership mismatch")
    captions = [by_token[t][0] for t in tokens]
    members = [by_token[t][1] for t in tokens]
    source_paths.append(original_path)
    if topology_path:
        source_paths.append(Path(topology_path))
    return {
        "members": members, "captions": captions,
        "source_hashes": {str(p.relative_to(root)): digest(p) for p in source_paths},
        "summary": {
            "split": split, "members": expected, "captions": len(captions),
            "visual_vectors_used": False, "structured_scope": "anchor_speed_acceleration_navigation",
            "topology_members": len(topology), "source_format": "finalized_jsonl", "native_fields_unchanged": True,
            "caption_models": dict(Counter(c.get("model", "unknown") for c in captions)),
            "caption_status": "machine_draft_not_fact_certified",
        },
    }


def load_inputs(config, root, output):
    corpora = {}
    started = time.monotonic()
    for split, count in config["expected"].items():
        save_json(output / "progress.json", {"status": "loading", "split": split})
        topology = root / config["topology_directory"] / f"{split}.jsonl" if config.get("topology_directory") else None
        corpora[split] = load_corpus(root, split, count, topology)
    if set(m["token"] for m in corpora["navtrain"]["members"]) & set(m["token"] for m in corpora["navtest"]["members"]):
        raise ValueError("Train/test token overlap")
    save_json(output / "inventory.json", {"schema_version": 1, "status": "verified", "splits": {s: c["summary"] for s, c in corpora.items()}, "source_hashes": {s: c["source_hashes"] for s, c in corpora.items()}, "elapsed_seconds": time.monotonic() - started})
    return corpora


def visible_tags(caption):
    descriptions = " ".join(a["description"] for a in caption["attributes"] if a.get("certainty") == "visible")
    return [t for t in caption["retrieval_tags"] if t in descriptions]


def caption_text(caption):
    """Use visible attribute claims, not uncertain prose or future supervision."""
    visible = [a for a in caption["attributes"] if a.get("certainty") == "visible"]
    # Tags remain draft claims. Only retain tags supported lexically by visible fields.
    descriptions = " ".join(a["description"] for a in visible)
    tags = visible_tags(caption)
    text = " ".join([descriptions, " ".join(tags) * 2])
    return re.sub(r"(?:前|后|左|右)(?:相机|视图|画面)|CAM_[FLRB]0|可见|画面中", "", text)


def fact_matrix(members):
    values = np.array([[m.get("features", {}).get(k, np.nan) for k in FACTS] for m in members], dtype=np.float32)
    commands = np.zeros((len(members), 5), dtype=np.float32)
    for i, m in enumerate(members):
        c = m.get("features", {}).get("nav_command_slot")
        commands[i, c if isinstance(c, int) and 0 <= c < 4 else 4] = 1
    return values, commands


def fit_representation(train, config):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    text = [caption_text(c) for c in train["captions"]]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3, max_df=0.85,
                                 max_features=config["max_text_features"], sublinear_tf=True, dtype=np.float32)
    sparse = vectorizer.fit_transform(text)
    svd = TruncatedSVD(n_components=min(config["text_dimensions"], sparse.shape[1] - 1), n_iter=3, random_state=config["seed"])
    svd.fit(sparse)
    facts, _ = fact_matrix(train["members"])
    mean = np.nanmedian(np.where(np.isfinite(facts), facts, np.nan), axis=0)
    mean = np.nan_to_num(mean)
    std = np.nanstd(np.where(np.isfinite(facts), facts, np.nan), axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1)
    return {"text": vectorizer, "svd": svd, "fact_center": mean, "fact_scale": std,
            "weights": config["weights"], "fit_split": "navtrain"}


def transform(corpus, model):
    from sklearn.preprocessing import normalize

    text = model["svd"].transform(model["text"].transform([caption_text(c) for c in corpus["captions"]]))
    facts, commands = fact_matrix(corpus["members"])
    valid = np.isfinite(facts)
    facts = np.where(valid, facts, model["fact_center"])
    facts = np.clip((facts - model["fact_center"]) / model["fact_scale"], -4, 4)
    facts = np.concatenate([facts, (~valid).astype(np.float32), commands], axis=1)
    blocks = [normalize(a).astype(np.float32) for a in (text, facts)]
    weights = [model["weights"][k] for k in ("text", "facts")]
    return np.concatenate([a * np.sqrt(w) for a, w in zip(blocks, weights)], axis=1), blocks


def fit_hierarchy(x, config, on_progress=None):
    """Train-only binary splits. Stored split centers reproduce training routes."""
    from sklearn.cluster import KMeans

    nodes, groups, centers, split_centers = [], {}, [], {}
    queue = [(np.arange(len(x)), -1, 0)]
    rng = np.random.default_rng(config["seed"])
    while queue:
        indices, parent, depth = queue.pop(0)
        node_id = len(nodes)
        node = {"id": node_id, "parent": parent, "depth": depth, "children": [], "train_count": len(indices)}
        nodes.append(node)
        groups[node_id] = indices
        if parent >= 0:
            nodes[parent]["children"].append(node_id)
        center = x[indices].mean(axis=0)
        centers.append(center)
        reason = "target_size" if len(indices) <= config["target_leaf_size"] else None
        if depth >= config["max_depth"]:
            reason = "depth_limit"
        if reason is None:
            sample = rng.choice(indices, min(len(indices), config["split_fit_samples"]), replace=False)
            if float(np.var(x[sample], axis=0).sum()) < 1e-9:
                reason = "identical_representation"
            else:
                estimator = KMeans(n_clusters=2, n_init=3, max_iter=30, random_state=config["seed"] + node_id, algorithm="lloyd").fit(x[sample])
                decisions = nearest(x[indices], estimator.cluster_centers_)
                children = [indices[decisions == side] for side in (0, 1)]
                if min(map(len, children)) < config["min_child_size"]:
                    reason = "small_child_preserved"
                else:
                    split_centers[node_id] = estimator.cluster_centers_.astype(np.float32)
                    queue.extend((child, node_id, depth + 1) for child in children)
        if reason:
            node["stop_reason"] = reason
        if on_progress and node_id % 100 == 0:
            on_progress({"status": "fitting_hierarchy", "nodes": len(nodes), "pending_nodes": len(queue)})
    return nodes, groups, np.array(centers), split_centers


def nearest(x, centers):
    distances = np.maximum((x * x).sum(1)[:, None] + (centers * centers).sum(1)[None, :] - 2 * x @ centers.T, 0)
    return np.argmin(distances, axis=1)


def assign_hierarchy(x, nodes, split_centers):
    result = np.zeros(len(x), dtype=np.int32)
    groups = {0: np.arange(len(x))}
    for node in nodes:
        indices = groups[node["id"]]
        if node["children"]:
            sides = nearest(x[indices], split_centers[node["id"]])
            for side, child in enumerate(node["children"]):
                groups[child] = indices[sides == side]
        else:
            result[indices] = node["id"]
    return result, groups


def ancestor(node_id, nodes, depth):
    while nodes[node_id]["depth"] > depth:
        node_id = nodes[node_id]["parent"]
    return node_id


def nearest_leaves(x, centers, leaves, k=3):
    ids, ds = [], []
    for start in range(0, len(x), 256):
        z = x[start:start + 256]
        d = np.maximum((z * z).sum(1)[:, None] + (centers * centers).sum(1)[None, :] - 2 * z @ centers.T, 0)
        ix = np.argsort(d, axis=1, kind="stable")[:, :k]
        ids.append(np.asarray(leaves)[ix])
        ds.append(np.sqrt(np.take_along_axis(d, ix, axis=1)))
    return np.concatenate(ids), np.concatenate(ds)


def export_members(path, corpus, assignment, nodes, x, centers, thresholds, config, alternatives=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    flagged = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for i, (m, c) in enumerate(zip(corpus["members"], corpus["captions"])):
            leaf = int(assignment[i])
            distance = float(np.linalg.norm(x[i] - centers[leaf]))
            outlier = distance > thresholds[leaf] + 1e-7
            flagged += outlier
            row = {"token": m["token"], "split": c["split"], "source_log": m["source_log"],
                   "leaf": leaf, "levels": {k: ancestor(leaf, nodes, d) for k, d in config["cuts"].items()},
                   "distance": distance, "out_of_train_radius": outlier,
                   "history_continuous": m.get("history_continuous", False),
                   "map_labels": m.get("map_labels", []), "lane_association": m.get("lane_association", "unknown"),
                   "source_jsonl": m.get("source_jsonl"), "source_line": m.get("source_line"),
                   "caption_model": c["model"], "caption_review": c.get("review_status"),
                   "use": "exploratory_cluster_member", "training_admission_granted": False}
            if alternatives is not None:
                row["nearest_training_leaves"] = [{"leaf": int(k), "distance": float(d)} for k, d in zip(alternatives[0][i], alternatives[1][i])]
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)
    return flagged


def summarize_nodes(corpora, x, nodes, groups, test_groups, centers, config):
    from sklearn.decomposition import PCA

    train = corpora["navtrain"]
    global_tags = Counter(t for c in train["captions"] for t in set(visible_tags(c)))
    projection = PCA(n_components=2, random_state=config["seed"]).fit(x)
    xy = projection.transform(centers)
    summaries, thresholds = [], {}
    for node in nodes:
        indices = groups[node["id"]]
        distances = np.linalg.norm(x[indices] - centers[node["id"]], axis=1)
        thresholds[node["id"]] = float(np.quantile(distances, config["outlier_quantile"]))
        tags = Counter(t for i in indices for t in set(visible_tags(train["captions"][i])))
        map_tags = Counter(t for i in indices for t in train["members"][i].get("map_labels", []))
        ranked = sorted(tags, key=lambda t: (-tags[t] * np.log1p(len(x) / global_tags[t]), t))[:8]
        speed = [float(train["captions"][i]["anchor_speed_kmh"]) for i in indices if isinstance(train["captions"][i].get("anchor_speed_kmh"), (float, int))]
        reps, seen = [], set()
        for local in np.argsort(distances, kind="stable"):
            i = indices[local]
            m, c = train["members"][i], train["captions"][i]
            if m["source_group"] in seen:
                continue
            seen.add(m["source_group"])
            reps.append({"split": "navtrain", "token": m["token"], "description": c["scene_description_zh"],
                         "tags": visible_tags(c), "speed_kmh": c.get("anchor_speed_kmh"),
                         "navigation_command": c.get("navigation_command"), "caption_status": "machine_draft"})
            reps[-1]["map_labels"] = m.get("map_labels", [])
            if len(reps) == 2:
                break
        summaries.append({**node, "label": " · ".join(ranked[:3]) or "待命名场景组",
                          "tags": [{"text": t, "count": tags[t]} for t in ranked],
                          "map_tags": [{"text": t, "count": count} for t, count in map_tags.most_common()],
                          "test_count": len(test_groups[node["id"]]),
                          "source_groups": len(set(train["members"][i]["source_group"] for i in indices)),
                          "radius_p99": thresholds[node["id"]], "mean_distance": float(distances.mean()),
                          "speed_mean_kmh": float(np.mean(speed)) if speed else None,
                          "speed_range_kmh": [float(min(speed)), float(max(speed))] if speed else None,
                          "xy": xy[node["id"]].tolist(), "representatives": reps})
    return summaries, thresholds


def run(config, corpora, root, output):
    import joblib
    from sklearn.metrics import adjusted_rand_score
    from threadpoolctl import threadpool_limits

    started = time.monotonic()
    def progress(v):
        save_json(output / "progress.json", {**v, "elapsed_seconds": time.monotonic() - started})
    with threadpool_limits(limits=config["threads"]):
        progress({"status": "fitting_train_representation"})
        model = fit_representation(corpora["navtrain"], config)
        joblib.dump(model, output / "representation.joblib")
        arrays, blocks = {}, {}
        for split in corpora:
            progress({"status": "transforming", "split": split})
            arrays[split], blocks[split] = transform(corpora[split], model)
            np.save(output / f"{split}-representation.npy", arrays[split])
        nodes, groups, centers, split_centers = fit_hierarchy(arrays["navtrain"], config, progress)
        assignments, test_groups = assign_hierarchy(arrays["navtest"], nodes, split_centers)
        train_assignment, train_route = assign_hierarchy(arrays["navtrain"], nodes, split_centers)
        if any(not np.array_equal(groups[n], train_route[n]) for n in groups):
            raise ValueError("Training split replay differs")
        np.savez_compressed(output / "hierarchy.npz", centers=centers, split_node_ids=np.array(list(split_centers)), split_centers=np.array(list(split_centers.values())))
        save_json(output / "hierarchy.json", {"schema_version": 1, "fit_split": "navtrain", "config": config, "nodes": nodes})
        progress({"status": "summarizing", "nodes": len(nodes)})
        summaries, thresholds = summarize_nodes(corpora, arrays["navtrain"], nodes, groups, test_groups, centers, config)
        leaves = [n["id"] for n in nodes if not n["children"]]
        alternatives = nearest_leaves(arrays["navtest"], centers[leaves], leaves)
        flags = {}
        for split, a in (("navtrain", train_assignment), ("navtest", assignments)):
            flags[split] = export_members(output / split / "members.jsonl", corpora[split], a, nodes, arrays[split], centers, thresholds, config, alternatives if split == "navtest" else None)
        # Sensitivity measures representational dependence, not semantic accuracy.
        sample = np.random.default_rng(config["seed"]).choice(len(train_assignment), min(2048, len(train_assignment)), replace=False)
        fact_center = np.array([blocks["navtrain"][1][groups[n]].mean(0) for n in leaves])
        text_center = np.array([blocks["navtrain"][0][groups[n]].mean(0) for n in leaves])
        agreement = {}
        for name, feature, prototype in (("text_only", blocks["navtrain"][0], text_center), ("facts_only", blocks["navtrain"][1], fact_center)):
            labels = np.asarray(leaves)[nearest(feature[sample], prototype)]
            agreement[name] = {"nearest_leaf_agreement": float(np.mean(labels == train_assignment[sample])), "adjusted_rand": float(adjusted_rand_score(train_assignment[sample], labels)), "scope": "2048_train_sample_same_leaf_prototypes_not_refitted_ablation"}
    sizes = [n["train_count"] for n in nodes if not n["children"]]
    level_counts = {k: len(set(ancestor(int(n), nodes, d) for n in leaves)) for k, d in config["cuts"].items()}
    hashes = {p: h for c in corpora.values() for p, h in c["source_hashes"].items()}
    changed = [p for p, h in hashes.items() if digest(root / p) != h]
    if changed:
        raise ValueError("Input changed during clustering; refusing complete status")
    summary = {"schema_version": 1, "status": "complete", "fit_split": "navtrain",
               "members": {s: len(c["members"]) for s, c in corpora.items()},
               "leaf_clusters": len(leaves), "tree_nodes": len(nodes), "level_clusters": level_counts,
               "leaf_size": {"min": min(sizes), "median": float(np.median(sizes)), "max": max(sizes), "p90": float(np.quantile(sizes, .9))},
               "stop_reasons": dict(Counter(n["stop_reason"] for n in nodes if not n["children"])),
               "test_occupied_leaves": len(set(map(int, assignments))), "out_of_train_radius": flags,
               "single_source_leaves": sum(s["source_groups"] == 1 for s in summaries if not s["children"]),
               "sensitivity": agreement, "source_files_unchanged": True, "training_route_replay_equal": True,
               "api_calls": 0, "student_scoring_calls": 0, "elapsed_seconds": time.monotonic() - started,
               "caption_quality": "machine_drafts_not_fact_certified", "retrieval_recall": None,
               "information_scope": "anchor_caption_speed_acceleration_navigation_map_topology_no_image_vectors_no_future",
               "input_format": "finalized_per_scene_jsonl", "topology_members": {s: c["summary"]["topology_members"] for s, c in corpora.items()},
               "image_vectors_used": False,
               "projection": "train_fitted_PCA_2d_for_browsing_only"}
    save_json(output / "navtrain/pools.json", {"schema_version": 1, "split": "navtrain", "pools": {str(n): [corpora["navtrain"]["members"][i]["token"] for i in groups[n]] for n in leaves}})
    save_json(output / "summary.json", summary)
    save_json(output / "clusters.json", {"schema_version": 1, "summary": summary, "cuts": config["cuts"], "nodes": summaries})
    save_json(output / "progress.json", {"status": "complete", "summary": "summary.json", "elapsed_seconds": time.monotonic() - started})
    return summary


def refresh_names(root, output):
    """Rebuild names from unchanged evidence without refitting or paid calls."""
    inventory = json.loads((output / "inventory.json").read_text())
    for relative, expected in inventory["source_hashes"]["navtrain"].items():
        if digest(root / relative) != expected:
            raise ValueError("Source changed since fit; cannot refresh names")
    payload = json.loads((output / "clusters.json").read_text())
    nodes = payload["nodes"]
    tags, global_tags = {}, Counter()
    for path in sorted((root / "lib/cluster_caption/navtrain/shards").glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                caption = json.loads(line)["cluster_caption"]
                token = caption["token"]
                if token in tags or caption["split"] != "navtrain":
                    raise ValueError("Duplicate or cross-split naming evidence")
                tags[token] = set(visible_tags(caption))
                global_tags.update(tags[token])
    counts = [Counter() for _ in nodes]
    pool = {}
    seen = set()
    with (output / "navtrain/members.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            seen.add(row["token"])
            pool.setdefault(str(row["leaf"]), []).append(row["token"])
            current = row["leaf"]
            while current >= 0:
                counts[current].update(tags[row["token"]])
                current = nodes[current]["parent"]
    if seen != set(tags):
        raise ValueError("Naming and membership coverage differ")
    for n in nodes:
        count = counts[n["id"]]
        ranked = sorted(count, key=lambda t: (-count[t] * np.log1p(len(tags) / global_tags[t]), t))[:8]
        n["label"] = " · ".join(ranked[:3]) or "待命名场景组"
        n["tags"] = [{"text": t, "count": count[t]} for t in ranked]
        for rep in n["representatives"]:
            rep["tags"] = sorted(tags[rep["token"]])
    summary = json.loads((output / "summary.json").read_text())
    summary["naming_policy"] = "tags_lexically_supported_by_visible_attributes"
    payload["summary"] = summary
    save_json(output / "clusters.json", payload)
    save_json(output / "summary.json", summary)
    save_json(output / "navtrain/pools.json", {"schema_version": 1, "split": "navtrain", "pools": pool})
    return {"renamed_nodes": len(nodes), "unchanged_members": len(tags), "api_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--refresh-names-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.refresh_names_only:
        print(json.dumps(refresh_names(root, args.output)))
        return
    corpora = load_inputs(config, root, args.output)
    if not args.inventory_only:
        print(json.dumps(run(config, corpora, root, args.output), ensure_ascii=False))
    else:
        print(json.dumps({s: c["summary"] for s, c in corpora.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
