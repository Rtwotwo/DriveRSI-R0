# NAVSIM scene catalog and visual observations

The catalog is a retrieval service for DriveHarness, not a learned RSI controller.
It stores source facts, unverified visual observations, and measured student
scores separately. A caption match is neither a causal diagnosis nor synthesis
approval. Model and scoring implementations are unchanged.

## Data boundaries

`navtrain.sqlite` contains the complete official navtrain population and can
serve training retrieval. `navtest.sqlite` contains the complete official
navtest population for evaluation-side inspection only. Database split is
immutable; `--purpose training` rejects a test catalog. The Harness provider
additionally checks canonical navtrain token and log membership. No test-image
embedding or caption query is automatically transferred to training retrieval.

Use the current operator's existing full-population indexes rather than
recomputing predictions. Import verifies official filter hashes, exact token
coverage, source log membership and source-file identity through the existing
ChainFlow benchmark reader. Original trusted local NAVSIM pickle files are the
only supported source format; do not import untrusted pickle files.

The annotator sees **four real history frames at 2 Hz, front/left/right cameras**.
The final history frame is the anchor. It never sees future frames, score,
student prediction, token or source-log name in its prompt. Missing or irregular
history is rejected rather than filled. Input evidence records image-byte SHA256,
camera, timestamp, actual downsampled size and original resolution.

## Commands (run from DriveRSI on 32341)

```bash
.venv/bin/python scripts/data/manage_scene_catalog.py import \
  --database exp/scene-catalog-v1/navtrain.sqlite --split navtrain \
  --index exp/navtrain-navtest-campaign-v1/navtrain/index.json \
  --descriptors exp/navtrain-navtest-campaign-v1/navtrain-scene-index.json

.venv/bin/python scripts/data/manage_scene_catalog.py import \
  --database exp/scene-catalog-v1/navtest.sqlite --split navtest \
  --index exp/navtrain-navtest-campaign-v1/navtest/index.json \
  --descriptors exp/navtrain-navtest-campaign-v1/navtest-descriptors.json

CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venvs/augustus/bin/python scripts/data/manage_scene_catalog.py label \
  --database exp/scene-catalog-v1/navtrain.sqlite --split navtrain \
  --config src/config/data/scene_catalog/qwen35_2b.json \
  --tokens exp/scene-catalog-v1/navtrain-pilot-tokens.json \
  --output exp/scene-catalog-v1/label-run --limit 8 --max-seconds 1200

.venv/bin/python scripts/data/manage_scene_catalog.py query \
  --database exp/scene-catalog-v1/navtrain.sqlite --split navtrain \
  --purpose training --tag traffic_dense --limit 8
```

Allocate an idle GPU before running the example. The `label` command is bounded,
not an automatic full-population annotation launch. `max_seconds` stops admission
of new scenes between attempts; model initialization and one in-flight generation
can exceed it. Generation has its own soft time and hard token cap. Use the
Harness process owner for an external hard process timeout if needed.

For visual tag/text queries, specify the profile ID returned by a label run:
`--profile <id> --visual-tag vehicle_ahead_visible` or `--text vehicle`.
The `query --text` command is literal FTS phrase matching. The separate `neighbors`
command below provides image-vector similarity; there is no assumed text/image
shared embedding space. Different model,
prompt, runtime or image-content versions are not silently merged. Resume skips
successful schema-validated annotations only when profile and evidence match.
Failures remain in `attempts` with raw output and costs. Schema validation does
**not** verify factual correctness; human spot checks are still needed.

Retrieval uses only the **latest attempt per scene and profile**. A failed latest
attempt hides an earlier label, and an explicit rejected visual review removes
that label from visual retrieval. Historical attempts remain auditable.
`stats().labeled_scenes` counts any historical schema-valid annotation, including
later rejected ones; it is not a quality-cleared or currently eligible count.
Image evidence describes the recorded catalog snapshot. A later execution stage
must verify the referenced image hashes before consuming mutable source files.
All requested filters are ANDed; results use deterministic token order and a
limit, not a relevance or lowest-score ranking.

Existing training scores can be imported with `import-scores --score-shard ...`
(repeat for each shard). The importer checks the original training-only run
identity, checkpoint/protocol agreement, population index and source log. It
never reruns the model. Scores use the native range **0–1**; query with
`--score-run <id> --score-below 0.95`. Score imports append in short transactions;
an interrupted import can be repeated and existing conflicting scores fail.

## Harness contract

Provider `navsim_scene_catalog_retrieval` accepts `database`, `structured_tags`,
`visual_tags`, `profile`, `text`, `limit`, `score_run`, `score_below` (nullable
optional filters). It freezes the selected training records, descriptor, observed
tags and measured score provenance into `navsim_scene_catalog_candidates`.
Later database changes do not silently change an already prepared experiment.

The next stages must still verify checkpoint freshness and current student
failure, perform same-anchor platform/model controls, and then generate and admit
the physical branch. This change does not launch synthesis or training.

## Models and limitations

The actual general annotator is the public Qwen3.5-2B model, obtained from the
[Qwen publisher's distribution](https://modelscope.cn/models/Qwen/Qwen3.5-2B)
with per-file revision, size and SHA256 verification. No external image API or
API credential is required. Keep weight files under `weights/vlm/`.

An experimental AUGUstus SFT backbone adapter strictly loads the existing native
checkpoint and applies its native time-zero text conditioning. A real pilot
returned task-specific future-object text rather than the requested observation
schema; that output was rejected. This backend is not recommended for bulk
captioning and does not imply any driving-performance result.

Current tags describe visible road layout, markings, participants, occlusion and
lighting. They cannot establish a collision cause, legal right of way, hidden
objects, precise speed or a safe correction. Labels are sparse: omitted tags are
unknown, not absent. Three selected cameras and reduced resolution can miss small
or rear objects. Full-population metadata does not imply full-population VLM
annotation. Validate annotation quality and throughput before scaling.

This is one prerequisite for RSI: searchable evidence. A useful next experiment
compares fixed sample selection with evidence-driven selection under the same
parent model and total budget, and feeds measured student benefit back into the
selection policy. Neither prompt edits nor labels alone prove recursive learning.

## Native image vector catalog

The `embed` command dumps frozen **Qwen3.5-2B visual encoder** features, bypassing
text decoding. The native post-merger patch features are averaged per image and
L2-normalized. All 12 vectors (4 history frames x 3 ordered cameras) are retained
as float16; the normalized mean scene vector is float32. Feature extraction loads
only the strictly matched visual checkpoint module. It does not change any
ChainFlow, DriveRL or AUGUstus driving weights or training loss.

This is a retrieval baseline, not a contrastively calibrated embedding model.
Sky, city appearance and camera pose can dominate similarity. Caption details,
structured geometry and measured model scores remain separate filters/evidence.
Never interpret a high cosine as the same failure cause.

```bash
CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venvs/augustus/bin/python scripts/data/manage_scene_catalog.py embed \
  --database exp/scene-catalog-v1/navtrain.sqlite --split navtrain \
  --config src/config/data/scene_catalog/qwen35_features.json \
  --tokens exp/scene-vectors-v2/navtrain-feature-tokens.json \
  --output exp/scene-vectors-v2/my-embedding-run --limit 64 --max-seconds 1800

.venv/bin/python scripts/data/manage_scene_catalog.py neighbors \
  --database exp/scene-catalog-v1/navtrain.sqlite --split navtrain \
  --purpose training --query-token <navtrain-token-with-feature> \
  --vector-profile <profile-from-embed-report> --limit 10 --shortlist 128 \
  --output exp/scene-vectors-v2/neighbors.json
```

Training queries accept only exemplars already in the **same navtrain database**.
Evaluation queries use a navtest database and `--purpose evaluation`. There is
no arbitrary query-vector injection or implicit test-to-train image retrieval.
The vector profile includes checkpoint, processor files, source, runtime, pooling
and layout; incompatible profiles cannot silently mix.

The current store is SQLite plus chunked exact cosine scanning, with 512 rows per
scan chunk. It does not require an external vector database or ANN service.
Coarse ranking uses the scene vector; a bounded shortlist is reranked by mean
cosine over corresponding camera/time image vectors. It is **not** an exact global
nearest-neighbor search under the rerank metric. The query token and its entire
source group are excluded by default. `--include-source` permits same-source
candidates, but still excludes self. Default diversity chooses the best coarse
representative per source group **before** reranking; it does not optimize over
all anchors of that group.

Structured tags, an explicit visual annotation profile, literal caption phrase,
and named training score-run threshold can be ANDed **before** ranking. Rejected
or superseded labels cannot satisfy a visual filter. Feature-only retrieval does
not imply that a missing/rejected caption has been accepted.

Successful dump rows are reused only for matching evidence and extraction
profile. Failed attempts are retained in job receipts and replace the current
vector entry with `failed`, so an obsolete successful vector cannot survive a
failed re-extraction. Search uses the stored snapshot; before synthesis, recheck
its image evidence against the source files. `max_seconds` is a between-scene soft
admission limit, not a hard GPU timeout. Split full populations into explicit
batches of at most 10,000 tokens; label batches remain capped at 1,000. Resume is
per token, so changing batch boundaries does not require re-encoding completed
matching entries. Keep all job receipts under `exp/`.

For 103,288 scenes the raw vectors alone are approximately 5.92 GB (12 x 2048 x
2 bytes plus 2048 x 4 bytes per scene), before SQLite/evidence overhead. Full-scale
latency, indexing and recall still require measurement; small-pilot timings must
not be presented as full-corpus throughput.

## Discriminative observation prompt

`src/config/data/scene_catalog/qwen35_retrieval_v2.json` selects `retrieval_v2`.
The original prompt remains available with `prompt_version: v1` (or omission).
The new prompt prioritizes visible bends/branches/merges, actor placement,
barriers, occlusion and distinctive context instead of generic road/weather
sentences. It asks for 3-6 supported observations (maximum 8), image references,
a consistent caption, and explicit unknowns. A claimed temporal change needs
both times as evidence. Side-camera image-left must not be confused with ego-left.
Do not force artificial differences between similar scenes. Format validation
cannot enforce factual or temporal grounding; retain manual quality review.

Labels, source facts and student scores remain distinct. Each prompt version
produces a separate profile; earlier rejected labels are not overwritten or
silently converted to approved observations. New labels are **unreviewed** until
checked, even if a previous prompt was rejected on the same token.

## Ranked retrieval in DriveHarness

The existing `navsim_scene_catalog_retrieval` provider accepts an optional block:

```json
{
  "vector": {
    "query_token": "a measured navtrain example",
    "profile": "embedding profile id",
    "shortlist": 128,
    "exclude_source": true,
    "diverse_sources": true
  }
}
```

All earlier keys remain required. Without `vector` its behavior is unchanged.
With it, the provider freezes ranked candidates, query/candidate image evidence,
encoder identity, filters, similarity details and any requested scores/labels.
Both query and candidates must pass official navtrain membership checks. Output
is still a candidate artifact, not synthesis approval. Downstream student
confirmation, cause diagnosis and data admission remain required.

The initial verbose `retrieval_v2` prompt is retained for comparison. It produced
only 6/16 schema-valid training responses in its pilot, including missing fields
and repeated tags. `qwen35_retrieval_v3.json` is the compact revision: current h3
images, at most four distinct tags, short spatial caption, unknowns first and a
schema reminder after the images. It avoids requesting guessed motion. The
shared validator still permits up to eight observations across versions; those
prompt limits are instructions, not proof of output truth. Neither version is
automatically a verified label source.

## Full-population batch preparation

Create a complete resumable job plan without launching all annotation work:

```bash
.venv/bin/python scripts/data/manage_scene_catalog.py plan \
  --database exp/scene-catalog-v1/navtrain.sqlite --split navtrain \
  --purpose training --job embed --config src/config/data/scene_catalog/qwen35_features.json \
  --batch-size 512 --output exp/scene-vectors-v2/full-navtrain-feature-plan
```

Use a fresh output directory. For navtest set both `--split navtest` and
`--purpose evaluation`, with its separate database. `--job label` uses the selected
caption config. The generated `plan.json` contains argv lists, not shell command
strings. Run batches with the allocated model environment and inspect each
receipt's `complete` and failures; a zero process exit does not mean all planned
samples were completed. Do not run concurrent writers to the same split; use
separate split jobs/GPU allocation. The plan itself is explicitly not execution.

The v3 pilot often returned a scalar image ID instead of the requested array.
`qwen35_retrieval_v4.json` therefore declares a **single current h3 image ID string**
per observation. The parser losslessly wraps that validated ID in the canonical
DB list. It does not supply missing fields, repair unknown IDs, or accept old v3
responses retrospectively. It enforces max four tags for this wire format; generic
canonical storage remains max eight. Historical failed profiles are retained.

`qwen35_retrieval_v5.json` is the latest pilot config. It keeps the v4 prompt and
accepts only losslessly equivalent singleton containers: an unknown string becomes
a one-item list, and a one-item evidence list becomes the declared single current
reference before canonical validation. This does not fill absent fields, merge
duplicate tags, accept historical/future IDs, or alter score/admission rules. Raw
answers and failed earlier profiles remain intact. Caption truth still needs review.

The final encoder loads checkpoint parameters with strict assignment and preserves
native FP32 rotary-frequency buffers. A whole-module BF16 cast in the first pilot
caused a nonzero native feature difference; that pilot is retained as a failure
record. A real 12-image comparison after correction had maximum absolute pooled
feature difference 0. Use the final recorded profile, not an earlier pilot profile.
This check concerns visual features, not driving-policy output or benchmark scores.


## Full trainval continuation and caption grounding

The official `scene_filter/navtrain` population is 103,288 anchors: 85,109 in the
native training-log partition plus 18,179 in the validation-log partition. The
native `navtrainval` recipe includes both; there is no missing extra population
outside the current catalog. Keep the existing official token namespace rather
than relabeling validation scenes as test. `navtest` stays in a separate DB.

`python scripts/data/run_scene_corpus.py <request.json>` runs complete feature
coverage before caption batches. The default serial path uses bounded CPU prefetch with one GPU encoder.
An optional `devices` list enables persistent independent GPU encoders with one
coordinator writing split-local databases; `workers` is the bounded CPU prefetch
count per GPU (1..8). It reuses the same
visual profile and validates evidence again on resume. Per-token failures count
as attempted, never as ready. Irregular history is retained as an explicit error.
Caption batches only consume ready image sources after both populations have been
attempted, and require a local matching `caption_release` observation-quality
record. A stopped batch can be resumed; failures do not silently become labels.

`qwen35_camera_v7.json` uses independent current-image calls for F0/L0/R0, with
references bound by code to the actual input camera. It returns small caption/tag
objects, then combines them with camera names. Historical frames are still kept
as source evidence but do not enter these current-image caption calls. No temporal
or intention inference is supported. The parser restricts common motion and fine
marking phrases, but its English word checks are neither exhaustive nor a factual
verifier. All accepted captions remain unverified observations. Earlier prompt
trials, including schema failures and example-echo failures, are retained.

`navsim_scene_expansion_plan` is the DriveHarness provider for measured training
weakness proposals. Config: `database`, `score_run`, `vector_profile`,
`per_category` (1..32), and `probes` (0..4 per category). It uses training-owned
scores, validates official membership, and selects distinct source groups per
collision/road/direction/interaction/progress/comfort category. All metrics use
strict <1 except progress <0.95. It exports low-scoring seeds and validates visual
neighbor retrieval separately; neighbors without failure evidence do not become
hard examples automatically. Source-history failures are routed to repair.

A reported same-metric neighbor is a measured proxy, not proof of the same cause
or human-rated retrieval accuracy. Both visual and random probes report measured
coverage explicitly. The output requests same-anchor diagnosis and eventual
synthesis; it does not bypass PDMS>=95, physical or visual admission. Test scene
IDs, images and per-scene scores are not read by this provider or exported.

## Hybrid visual and semantic retrieval

`navsim_scene_hybrid_retrieval` adds independent candidate routes, rather than
requiring text/label matches before visual ranking. Its config contains
`database`, `query_token`, `vector_profile`, `caption_profile` (null allowed),
`score_run`, `target_category` (one declared weakness category), `limit` (1..64),
and `pool` (limit..512, normally 128).

Both routes use a single read snapshot of ready training vectors. The visual route
uses the unchanged native pooled features and aligned-image cosine. The semantic
route uses inverse-frequency-weighted overlap of structural history conditions,
plus lexical BM25 for reviewed captions when available. Only a newest attempt that is validated
and whose latest review is `spot_check_pass` contributes; a newer failed or
unreviewed attempt suppresses earlier approved text; rejected/unreviewed text
is excluded even from BM25 document statistics. Caption BM25 is lexical English
retrieval, not a text embedding or a verified causal description.

Rank lists are combined with reciprocal rank fusion (constant 60); visual and
semantic routes have equal rank weight. Structured/caption sublists are fused
inside the semantic route. No student score determines a rank. This opt-in provider preserves visual top-K
and add semantic top-K candidates from new source groups (at most 2K); RRF
orders this retained union. Fused top-K is separately reported as a diagnostic
route because the initial real pilot lost some visual hits after truncation.
Compare the union against visual top-2K as well as visual top-K before claiming
a benefit, since its candidate budget can be larger. A query with no
semantic evidence explicitly falls back to visual retrieval. Source-group
exclusion and per-list diversity prevent adjacent anchors filling a shortlist.
Missing captions do not remove a scene from the visual or structured route.

After ranking, the provider attaches measurements from the declared training
checkpoint/run. A matching measured target weakness and continuous history yield
`synthesis_requested=true`; missing scores yield `needs_student_measurement`.
All candidates remain `trainable=false`, and require same-anchor diagnosis,
PDMS>=95 teacher execution, physical integrity and visual admission downstream.
The provider validates the whole ready population against official navtrain
membership and never accepts test exemplar details.

Current caption pilots are not released for bulk labeling. A real evaluation
without approved captions must be described as visual + structural fusion, not
as evidence of a caption benefit. Compare visual, semantic and fused routes on
the same frozen population with the same queries, top-k budget and source
exclusion. Report unscored neighbors separately; shared failed submetrics are
only a proxy, not recall against human-annotated causal relevance.

## Multi-GPU corpus extraction

Add `"devices": [6, 7]` to the full-corpus request to use independent native
vision encoders. Other GPU allocations are not changed. Both full split
populations are scheduled in alternating bounded chunks, preserving source-log
locality inside each split. Feature batch size, precision, preprocessing,
pooling and profile are unchanged. The coordinator holds both original feature
locks and is the only database writer. Workers receive split-tagged scene records
and image evidence; their output cannot be committed under another split.

Before handoff, send SIGTERM to the previous feature coordinator and wait for it
to exit. Existing vectors remain in the original databases. On resume, image
evidence is re-read and validated before ready rows are reused; failed histories
are explicitly retried and reported. A worker crash stops the run; committed
rows survive and uncommitted work is recomputed on restart. No bulk caption call
can begin until both populations are accounted for, and the existing matching
caption-quality release is still required. Feature completion with failed input
rows means complete accounting, not a complete ready vector population.

Top-level progress points to `parallel-features/progress.json`, which includes
per-split new/reused/failed counts, worker PIDs, timing and current status. These
per-pass counts must not be added to an older cumulative count. Exact current
coverage is `vectors WHERE profile=<current_profile> GROUP BY status` in each DB.
No navtest tokens or images belong in public progress summaries.


## Observed irregular-history vectors

Some catalog scenes have four decodable real records but nonuniform intervals.
The ordinary 2 Hz history loader still rejects them. For **visual retrieval only**,
`scripts/data/recover_scene_vectors.py <request.json>` extracts the original four
records with their actual timestamps; it does not synthesize, duplicate, interpolate,
or retimestamp images. Only explicitly listed original per-split scenes whose
base extraction failed with `Irregular history is not silently padded` are eligible.
Each request is bounded to 1,000 tokens per split and has this shape:

```json
{
  "databases": {"navtrain": "exp/scene-catalog-v1/navtrain.sqlite"},
  "tokens": {"navtrain": ["an explicitly audited training token"]},
  "feature_config": "src/config/data/scene_catalog/qwen35_features.json",
  "output": "exp/my-observed-history-recovery"
}
```

Run in the native feature environment with an explicitly allocated GPU. A separate
`--verify-resume` invocation revalidates source image evidence and requires all
rows to be reused without constructing an encoder. Training and test requests use
distinct databases. Detailed test receipts remain private; publish only counts.

The supplemental profile records its exact unchanged base encoder/profile, its
observed irregular sampling, and `training_history_repaired=false`. Original
ready vectors, base-profile failure records, log data and history-continuity flags
remain intact. Source or inference failures invalidate the supplemental row rather
than leaving an obsolete recovered vector ready.

For visual Harness retrieval, add `history_recovery_profile` alongside `vector` to
`navsim_scene_catalog_retrieval`. A connection-local temporary read view combines
compatible supplemental rows with base ready rows (base ready rows take priority).
Neither the stored base profile nor any permanent database view is changed. The
query/result provenance reports the actual storage profile and whether history
was irregular. `aligned_view_cosine` then compares **ordered camera/record views**,
not synchronized time offsets; `nominal_2hz_pair=false` makes that distinction
explicit. The hybrid provider and ordinary feature/caption commands retain their
existing strict behavior unless separately adapted; this option is not global.

The retrieved artifact is still `trainable=false` and `synthesis_authorized=false`.
Candidates with discontinuous histories are routed to `source_history_repair_required`.
A complete visual vector population therefore does **not** imply repaired 2 Hz
training sequences, verified labels, or improved hard-example retrieval accuracy.
