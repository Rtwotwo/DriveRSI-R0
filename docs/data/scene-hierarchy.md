# Scene caption and topology hierarchy

This is an exploratory, scene-level hierarchy over the finalized caption JSONL. It does not access `lib/retrieval` or reuse image similarity vectors. It does not score a student, invoke an API, grant training admission, or modify native AutoDrive-P3 targets.

Canonical data: `lib/cluster_caption/clusters/`.

| Artifact | Purpose |
| --- | --- |
| `navtrain/members.jsonl` | One training scene per row, fine leaf, ancestor cuts, original JSONL/line reference, distance and map conditions |
| `navtest/members.jsonl` | Separate private test members; frozen hierarchical assignment and three nearby training leaves |
| `navtrain/pools.json` | Training tokens grouped by leaf for downstream retrieval |
| `hierarchy.json`, `hierarchy.npz` | Shared tree and split centers |
| `representation.joblib` | Train-fitted text vocabulary, SVD and state normalization; trusted local artifact only |
| `clusters.json`, `summary.json` | Train-derived names/representatives and aggregate membership diagnostics |
| `topology/{navtrain,navtest}.jsonl` | Independent native log/map facts per anchor |
| `inventory.json` | Full coverage, original-field checks and source digests |

## Inputs and meanings

The input is `lib/cluster_caption/<split>/shards/*.jsonl`. Every row must match its original annotation key, scene and native fields. The native `solution` and future trajectories are checked for preservation but excluded from clustering. Current anchor speed, acceleration and navigation instruction are structured facts; a turn instruction is not evidence of an executed turn.

All visible dimension-specific caption descriptions are retained for text representation. Tags are included only when lexically supported by a visible attribute; uncertain attributes and unsupported tags remain in the original evidence but do not name the clusters. Captions are still unverified machine drafts. This filtering is an evidence boundary, not a factual accuracy certificate.

Map facts come from the original log's exact anchor token and global position, queried against the matching source map: intersection and connector-polygon containment, distances to intersections/crosswalks capped at 100 m, lane/connector counts within 30 m, and incoming/outgoing lane-graph degree. Graph degree is populated only for a unique ordinary-lane polygon match. An anchor within an intersection connector need not be on an ordinary-lane polygon; it is not forced onto a nearby lane. These features do not establish a maneuver, a right-of-way decision, or a causal interaction.

Train-only representation: Chinese character 2–4-gram TF-IDF (`min_df=3`, `max_df=0.85`, at most 40,000 features), randomized SVD to 128 dimensions; structured values use training medians and standard deviations, clipping to ±4, missingness and navigation indicators. Both blocks are L2 normalized per scene and scaled by square roots of 0.85 / 0.15. These weights are experimental defaults, not an established optimum. Text-derived vectors are separate from the existing image-vector database.

Train-only binary hierarchy: KMeans with two centers, up to 4,096 sampled members per split, three initializations and 30 iterations. All members route by Euclidean distance to those stored centers. Target leaf size is 64, minimum child size 8, maximum depth 20, seed 42. A split producing an undersized child is preserved as a larger leaf, not forcibly divided. Depth cuts at 5/8/11 support coarse, medium and fine browsing; final leaves provide the most detailed view.

The frozen training transformation and tree route every test member. Alternative nearest leaves are supplementary candidates, not calibrated probabilities or proof of equal failure causes. Distances beyond each training leaf's empirical 99th-percentile radius are marked uncertain. Small leaves mean this is not a global 1% false-positive guarantee. No relevance-labelled Recall@K is available.

## Run

Install the optional `clustering` dependency group. Map extraction also requires the `maps` group. Entry scripts are thin wrappers around `src/data/catalog/scene_topology.py` and `scene_hierarchy.py`.

```sh
.venv/bin/python scripts/data/extract_scene_topology.py \
  --index-root exp/navtrain-navtest-campaign-v1 \
  --maps-root /path/to/datasets/maps \
  --output lib/cluster_caption/clusters/topology --workers 6
.venv/bin/python scripts/data/cluster_scenes.py \
  --config src/config/data/scene-hierarchy.json \
  --output lib/cluster_caption/clusters
```

`--inventory-only` checks sources without fitting. `--refresh-names-only` refreshes names from unchanged source answers and rebuilds the training pool lookup; memberships and fitted models are not changed. No online automatic monitoring is started.

## Visualization and boundaries

The existing unified platform's `#view=clusters` has depth selection, tag/map-condition search, a 2D train-centroid PCA map, size sorting, drill-down and training representatives. The projection is for browsing; its 2D separation is not a retrieval score. Only training tokens/descriptions and aggregate test counts are published. Private test rows stay in the separate library. The browser grants no production indexing or training approval.

This first hierarchy organizes scene conditions. It is not a verified interaction taxonomy, a student failure classifier, or evidence that more clusters improve RSI training. Future refinement can use reviewed event labels and relevance judgements without repeating paid caption generation.
