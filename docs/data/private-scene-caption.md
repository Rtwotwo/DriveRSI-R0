# Private scene caption batches

This runner produces two separate artifacts from existing, verified **navtrain** whitebox bundles:

- `retrieval_history`: four past/current source timestamps, four cameras. Future observations are excluded.
- `video_clip`: four sampled timestamps across the full offline clip, four cameras. This output is not an online policy input or a history-only retrieval document.

Both outputs reuse `scene_description_v2` and its evidence-linked observations. The measured geometry remains authoritative. Text describes appearance; it cannot override object identity, trajectory, camera calibration, occlusion or signal state. Source annotation interpolation at0.1s does not become an actual10Hz camera measurement.

## Private configuration

All actual provider connection settings and credentials are supplied at runtime from `/path/to/private/`. The loaders require mode0600 and reject files outside that directory. No process-environment credential fallback is used by this caption runner. The generic example under `src/config/data/semantic_conditioning/` contains placeholders only and must be copied/configured outside the repository before use.

The two separate services currently have private files `bailian.env` / `caption-comparison.json` and `advisor.env` / `advisor.json`. Do not copy these into an experiment directory, include them in a release, or print their values. The older Harness advisor takes its key in the process environment; load that environment from its private file immediately before invoking the existing advisor, and pass the external configuration path. No credential is embedded in either client.

The caption client uses HTTPS directly, rejects redirects and public DNS addresses, and disables proxy/public fallback. Payloads contain resized source photos, their local evidence identifiers and a minimized whitebox description; they exclude source filenames, test scenes, student scores and target trajectories. Source paths are retained locally for audit, not sent to the model.

## Bounded entry point

Create an experiment manifest with `database` pointing to the navtrain catalog and `bundles` listing1–128 existing `source.json` bundle directories. The configuration is an external file. Run from the repository root:

```bash
.venv/bin/python scripts/data/run_private_caption.py prepare \
  --manifest exp/bailian-scene-caption-v1/manifest.json \
  --config /path/to/private/caption-comparison.json \
  --output exp/bailian-scene-caption-v1/new-batch

.venv/bin/python scripts/data/run_private_caption.py run \
  --manifest exp/bailian-scene-caption-v1/manifest.json \
  --config /path/to/private/caption-comparison.json \
  --env-file /path/to/private/bailian.env \
  --output exp/bailian-scene-caption-v1/new-batch
```

Use the identical command and directory for ordinary resume. Successful results and finalized failures are reused. A response received before interruption is reconstructed locally. An unresolved in-flight or gateway result is not automatically resent. Only HTTP429 is retried, within the attempt and lifetime request limits. Changed provider payloads are new jobs; implementation-only edits do not cause another payable request. Legacy adapter-hash batches remain read-only until explicitly reconciled. The batch lock prevents two runners from submitting the same output directory concurrently.

This is a bounded concurrent client, **not the provider's discounted Batch API**. A new output directory has its own request budget; do not create fresh directories to retry ambiguous work. Limits are requests, not a guaranteed currency cap. Provider failures without usage remain unknown billing outcomes.

## Caption quality and release

The fixed JSON asks for overview, road layout, surroundings, lighting, pavement, traffic arrangement, temporal changes and up to four distinctive features. Each claim references supplied image IDs. Unknown attributes stay unknown. Semantic whitebox colors are not paint colors; camera-image left/right is not automatically ego-frame left/right. Sparse images cannot establish all temporal behavior.

`format_valid_unreviewed` proves only schema/evidence-ID validity. Raw text and draft prompts remain separate from the reviewed description. A format-valid caption can still confuse a car with a minivan, invent crossing motion, swap paint colors, or infer an enclosed bridge from railings. Review and release claims individually through the existing v2 `approved_paths` interface. Video conditioning still requires separate geometry and temporal validation. Caption approval never grants training-data or renderer admission.

The small pilot and its calibration are recorded under `exp/bailian-scene-caption-v1/`. Two urban scenes do not establish dataset-wide accuracy, retrieval quality, or an automatic model ranking. Prices in comparison artifacts are published list-price calculations from returned usage, not the account invoice; failed or rejected calls must not be omitted from accounting.
