"""Train-fitted historical scene-context pools, not causal failure taxonomies."""

from collections import Counter

import numpy as np

from .store import identity

NUMERIC = (
    "ego_speed_mps", "history_speed_range_mps", "history_yaw_range_rad",
    "vehicle_count_30m", "pedestrian_count_30m", "bicycle_count_30m",
    "nearby_vehicle_count_15m", "nearby_static_count_15m", "static_count_30m",
)


def scene_facts(rows):
    """Fixed documented facts; absent values get a separate missing indicator."""
    values = np.zeros((len(rows), 2 * len(NUMERIC) + 5), dtype=np.float32)
    for i, row in enumerate(rows):
        if row.get("information_set") != "current_state_history":
            raise ValueError("Future-aware descriptors cannot enter history retrieval")
        facts = row["features"]
        for j, name in enumerate(NUMERIC):
            v = facts.get(name)
            if v is None:
                values[i, j + len(NUMERIC)] = 1
            elif type(v) not in (float, int) or not np.isfinite(v) or v < 0:
                raise ValueError("Invalid nonnegative scene fact: " + name)
            else:
                values[i, j] = np.log1p(v)
        command = facts.get("nav_command_slot")
        slot = command if type(command) is int and command in (0, 1, 2, 3) else 4
        values[i, 2 * len(NUMERIC) + slot] = 1
    return values


def squared_distances(x, centers):
    return np.maximum(
        (x * x).sum(1)[:, None] + (centers * centers).sum(1)[None, :] - 2 * x @ centers.T,
        0,
    )


def assign(x, centers, *, batch=2048):
    labels, distances = [], []
    for offset in range(0, len(x), batch):
        d = squared_distances(x[offset:offset + batch], centers)
        k = d.argmin(1)
        labels.append(k)
        distances.append(d[np.arange(len(k)), k])
    return np.concatenate(labels), np.concatenate(distances)


def transform(vectors, rows, model):
    if len(vectors) != len(rows) or vectors.shape[1] != model["projection"].shape[0]:
        raise ValueError("Scene/vector layout mismatch")
    visual = np.asarray(vectors) @ model["projection"]
    facts = scene_facts(rows)
    blocks = []
    for x, prefix in [(visual, "visual"), (facts, "facts")]:
        x = np.clip((x - model[prefix + "_mean"]) / model[prefix + "_scale"], -5, 5)
        blocks.append(x / np.sqrt(2 * x.shape[1]))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def fit(vectors, rows, *, split, clusters=128, fit_samples=8192, iterations=20, seed=42):
    if split != "navtrain":
        raise ValueError("Only navtrain may fit shared scene partitions")
    if len(rows) < clusters or clusters < 2 or iterations < 1 or fit_samples < clusters:
        raise ValueError("Invalid clustering budget")
    rng = np.random.default_rng(seed)
    dim = min(32, vectors.shape[1])
    projection = rng.standard_normal((vectors.shape[1], dim)).astype(np.float32) / np.sqrt(dim)
    visual = np.asarray(vectors) @ projection
    facts = scene_facts(rows)
    model = {"projection": projection}
    for x, prefix in [(visual, "visual"), (facts, "facts")]:
        model[prefix + "_mean"] = x.mean(0)
        model[prefix + "_scale"] = np.maximum(x.std(0), 1e-4)
    x = transform(vectors, rows, model)
    chosen = np.sort(rng.choice(len(x), min(fit_samples, len(x)), replace=False))
    sample = x[chosen]
    # Seeded k-means++ followed by bounded Lloyd iterations; no N x N graph.
    centers = [sample[rng.integers(len(sample))].copy()]
    nearest = squared_distances(sample, np.stack(centers))[:, 0]
    for _ in range(1, clusters):
        total = float(nearest.sum(dtype=np.float64))
        if total <= 1e-10:
            raise ValueError("Fewer distinct scene contexts than requested clusters")
        idx = rng.choice(len(sample), p=nearest.astype(np.float64) / total)
        centers.append(sample[idx].copy())
        nearest = np.minimum(nearest, squared_distances(sample, np.stack(centers[-1:]))[:, 0])
    centers = np.stack(centers)
    for _ in range(iterations):
        labels, distances = assign(sample, centers)
        for k in range(clusters):
            members = sample[labels == k]
            centers[k] = members.mean(0) if len(members) else sample[distances.argmax()]
    labels, distances = assign(x, centers)
    model["centers"] = centers
    model["outlier_distance"] = np.asarray(np.quantile(distances, 0.99))
    model["fit_indices"] = chosen
    return model, x, labels, distances


def cluster_summaries(rows, labels, distances, count):
    result = []
    for k in range(count):
        indices = np.flatnonzero(labels == k)
        terms = Counter(t for i in indices for t in set(rows[i].get("conditions", [])))
        signatures = Counter(identity(sorted(rows[i].get("conditions", []))) for i in indices)
        representatives = sorted(indices, key=lambda i: (float(distances[i]), rows[i]["token"]))[:5]
        boundary = sorted(indices, key=lambda i: (-float(distances[i]), rows[i]["token"]))[:5]
        result.append(dict(
            cluster=k, count=len(indices),
            source_groups=len({rows[i]["source_group"] for i in indices}),
            history_fact_terms=[dict(term=t, count=n) for t, n in terms.most_common(8)],
            fact_subgroups=len(signatures),
            representatives=[rows[i]["token"] for i in representatives],
            boundary=[rows[i]["token"] for i in boundary],
            causal_label=None, interaction_type="unverified", topology="not_in_descriptor",
        ))
    return result


def rank_pool(vectors, query, indices, rows, limit, *, per_source=None):
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return []
    # Bounded blocks avoid copying the full 0.85GB matrix per query.
    scores = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), 4096):
        part = indices[start:start + 4096]
        scores[start:start + len(part)] = vectors[part] @ query
    order = np.lexsort((indices, -scores))
    selected, counts = [], {}
    for pos in order:
        i = int(indices[pos])
        group = rows[i]["source_group"]
        if per_source is not None and counts.get(group, 0) >= per_source:
            continue
        counts[group] = counts.get(group, 0) + 1
        selected.append((i, float(scores[pos])))
        if len(selected) == limit:
            break
    return selected


def candidate_indices(context, centers, labels, probes):
    if not 1 <= probes <= len(centers):
        raise ValueError("Invalid probe budget")
    d = squared_distances(context[None, :], centers)[0]
    order = np.lexsort((np.arange(len(d)), d))[:probes]
    return np.flatnonzero(np.isin(labels, order)), order
