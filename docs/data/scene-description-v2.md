# Scene description v2 and downloadable source whiteboxes

The stable output is `driversi.scene_description.v2`, validated by `src/config/data/semantic_conditioning/scene_description_v2.schema.json`. Existing v1 appearance helpers remain available. `caption_semantic_conditions.py` now defaults to v2; `--schema-version 1` explicitly selects the old contract. Old results remain unchanged.

## JSON contract

| Key | Meaning |
|---|---|
| schema_version / scope | Fixed version and retrieval_history or video_clip |
| scene_identity | Training token/split and source_log or executed_branch; identifiers are not inserted into embedding prose |
| information_scope | -2..0s history or -2..+5s offline clip; not an online policy input |
| scene_overview | Deterministic whole-scene summary and optional reviewed visual overview |
| structured_facts | Nearby and all-anchor object counts, dimensions, relative positions, measured path/heading, timeline, palette, signals and limitations |
| appearance | VLM claims, exact image evidence, approved claim paths and explicit review |
| distinctive_features | Structured distinctions plus individually reviewed visual distinctions |
| retrieval | Past/current embedding text and numeric facets; null for future video scope |
| video_generation | Full-clip positive/negative prompts, per-camera appearance and geometry constraints; null for retrieval |
| evidence / facts_identity | Input image references and existing content identity |
| limitations / trainable | Explicit limits; description construction never grants RGB-training admission |

VLM output has seven fixed evidence-linked fields: overview, road_layout, surroundings, lighting, road_surface, traffic_configuration and temporal_changes, plus a list of distinctive_features. Each claim is `{text, evidence_ids}`. Unknown is exactly `{text: "unknown", evidence_ids: []}`. Evidence membership checks only demonstrate that referenced inputs exist, not that the claim is correct. The pilot requests35words,1–2evidenceIDs per claim and at most2distinctive features to fit the model's output budget. The parser accepts up to60words/16IDs/4distinctive claims as the published upper bound. Raw unsuccessful attempts remain recorded.

Whole-scene prose is not a list of imagined events. Structured summaries use all sampled states in the selected information scope, a1s timeline, anchor objects and nearest8object details. Counts within30m are distinct from full-frame annotation counts. Heading change is signed, path length sums successive XY displacements; neither implies intent. Temporal sampling remains source interpolation or executed physics as declared by the bundle.

Only explicitly reviewed appearance claims enter released text. A reviewer can approve `overview`, `lighting` or `distinctive_features.0` independently; an approved overview does not approve a wet-road claim. Camera-specific prompts use only reviewed claims whose evidence all comes from that camera; cross-camera claims stay in the global prompt. Descriptions never turn semantic class colors into paint colors, identify all generic vehicles as one subtype, or invent an object-to-photo identity match.

## Motion and download contract

`motion_audit.py` separately reads raw annotations and checks71states against source timestamps, same-ID positions, shortest-arc headings, source dimensions, instance continuity, calibration transport and presence semantics. It recomputes transforms without using the exporter's object resampler. Tests deliberately corrupt one frame's position, heading, identity, camera or timestamp and require rejection.

Agreement is with raw2Hz annotation and the declared interpolation rule, not with unobserved true10Hz motion. Dimensions/calibration are held from the preceding observation between frames. New labels do not appear early. Missing labels are held only until the next source sample with uncertainty recorded. Raw-log fidelity is not physical correctness of annotations. Executed branches cannot pass the original-log audit.

The complete source ZIP contains source/map/palette/manifest JSON, descriptions, motion audit,284semantic PNGs and four70-frame/10fps MP4s. The7.0s endpoint remains as PNG/JSON. No real camera photos or test samples are packed. Image hashes, render source identity, frame completeness and archive CRC are checked before publication. PNGs and MP4s are stored without redundant ZIP compression; JSON is compressed.

```bash
.venv/bin/python scripts/data/package_semantic_conditions.py \
  --bundle exp/scene-description-v2/bundles/TOKEN \
  --output exp/scene-description-v2/downloads/TOKEN.zip

CUDA_VISIBLE_DEVICES=7 .venvs/augustus/bin/python scripts/data/run_semantic_caption.py \
  --database exp/scene-catalog-v1/navtrain.sqlite \
  --bundle exp/scene-description-v2/bundles/TOKEN \
  --model weights/vlm/Qwen3.5-2B --schema-version 2 \
  --output exp/new-caption-attempt
```

This is a bounded condition-export and caption pilot. It does not certify generated video appearance, pixelwise occlusion, driving safety or model improvement. Road height and cuboid occlusion remain proxies; source-vs-branch geometry and future-scope separation remain mandatory.

The decoder may append exactly one missing final root brace when the complete strict schema then validates. Original raw text and the syntax-repair receipt are retained; no claim or evidence is invented. This does not grant factual approval. The example with released appearance uses explicitly labeled assistant visual corrections of a Qwen draft.
