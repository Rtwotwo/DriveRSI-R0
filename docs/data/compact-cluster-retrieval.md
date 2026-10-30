# Compact augmentation selection

Implementation: `src/rsi/experiments/compact_cluster_retrieval.py`.
Cost audit: `src/rsi/experiments/synthesis_cost_audit.py`.

The campaign freezes the existing ChainFlow 9405 evaluation and selects its 239
zero-score queries. Nonzero low-score queries are outside this first batch.
Training sources remain official navtrain members. No navtrain student inference
or score_run is required; no paid caption inference runs.

The text/state/topology hierarchy is not refitted with visual vectors. Each query
uses its existing primary leaf. All training members in that leaf are ranked by
the mean of twelve normalized, corresponding-image cosine scores from the frozen
visual collection. This is a second-stage visual ranking, not centroid-only
ranking. The original global visual Top16 is reused as a baseline.

For precision-oriented production selection, require the same map, anchor
distance <=25 m, yaw difference <=30 degrees, continuous training history, and a
query not marked outside the training cluster radius. Retain one candidate per
query. These are explicit selection conditions, not independent accuracy ground
truth after filtering; semantic and causal relevance remain unmeasured. Nearby
parallel lanes, different signal states and participant configurations can remain.

The compact budget greedily selects at most 24 training tokens by additional
query coverage, breaking ties by mean visual similarity and token. It permits one
seed per source log and stops when no new query can be covered. A separate small
queue applies the same conditions to registered GS-supported tokens. Neither
queue claims visual or training admission; assets are not substituted from
unrelated scenes to fill a quota. Query identities stay in private provenance.

The stricter gate trades coverage for a smaller workload. In this campaign,
ungated cluster-visual Top1 has worse geographic performance than ungated global
visual Top1. Do not describe clustering alone as a precision improvement.

## Invocation

From the existing server project and existing environment:

```sh
.venv/bin/python -m driversi.rsi.experiments.compact_cluster_retrieval \
  --project <repository> \
  --output <repository>/exp/chainflow9405-compact-retrieval-v1

.venv/bin/python -m driversi.rsi.experiments.synthesis_cost_audit \
  --campaign exp/chainflow-expansion24-v1/campaign \
  --output exp/chainflow9405-compact-retrieval-v1/cost-audit.json
```

Preserve completed rankings rather than repeating remote per-image-vector reads.
The experiment checks expected memberships, profile and input identities and
stores input hashes. Reuse the frozen result only for its declared source version;
new source data require a new experiment version.

## Timing interpretation

Report input loading, exact ranking and filtering separately. Time spent reading
original twelve-view BLOBs must not be confused with earlier cluster-ID lookups.
Teacher and render stage wall times describe service work, not GPU-only kernel
time or total campaign elapsed time. Reused receipts count only once.

The historical 259-attempt batch produced only four qualified samples under a
24 dB experimental quality profile, with 187 trajectory-duplicate rejections.
Its yield is not a prediction for a >25 dB run. Conditional cost scenarios assume
valid existing assets and do not guarantee twenty distinct samples per anchor.
Asset reconstruction, queue time, labeling/export, and training are additional.

The bounded new pilot reuses the existing continuous native 7-second DriveRL
skill and its trajectory admission profile. It does not enable training or alter
model/loss. Its background-contact evidence remains pending; passing teacher
PDMS alone does not publish a training sample.
