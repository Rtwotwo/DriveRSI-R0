# NAVSIM semantic whitebox and caption conditions

The condition bundle describes an original training log or one executed physical branch. It is not itself a qualified RGB training example. It does not change native policies, losses, PDMS, or test data ownership.

## Temporal contract

Each window spans **[-2,+5] seconds relative to its anchor**, with **71 states at 0.1-second intervals**, including both endpoints. Four planning history samples are -1.5,-1,-0.5,0; ten future points are +0.5 through +5. The extra -2-second state is a prelude. H.264 playback has 70 frames at10fps over [0,7); the endpoint remains in PNG and JSON.

NAVSIM source sampling is approximately2Hz. These are not71 real sensor captures. Same-ID positions and shortest-arc heading are bracket-interpolated; ego rotation uses SLERP. Gaps above0.75s and unavailable endpoints reject the window, without extrapolation. New actors appear at their first annotation. Disappearing actors are held only until the next source sample and tagged uncertain. This is annotation visibility, not a physical birth/death claim. An executed branch uses its actual0.1s physical states, presence and lifecycle. Up to1ms source-clock jitter is recorded explicitly.

## Portable bundle

| File | Meaning |
|---|---|
| source.json | Source-frame brackets, all objects, poses, dimensions, motion basis, cameras, timing and signal records |
| branch.json | Optional actual physical branch; never substituted with source motion |
| map.json | Original local map polygons/lines and native properties, layer availability and provenance |
| palette.json | Versioned class-color legend; stable instance IDs identify individual objects |
| retrieval-facts.json | Past/current-only facts, nearby counts and spatial relations |
| video-facts.json | Explicit offline full-clip facts; cannot enter current-policy inputs |
| branch-*-facts.json | Separate descriptions for the changed physical branch |
| manifest.json | Content hashes, exporter identity and completion receipt |

Vehicle means an unspecified vehicle, not a sedan, bus or van. Dimensions are supplied, but type is not guessed from dimensions. Blue means vehicle class, not blue paint. Pedestrians, bicycles, cones, barriers and signs have distinct class colors. Unknown subtypes remain unknown. Source actor IDs and integer instance IDs remain stable through the clip. No class-colored frame is claimed to contain verified appearance or per-pixel visibility.

All source map layers remain inJSON. Raster conditions show road surfaces, walkways, crosswalk polygons and annotation boundaries. Navigation centerlines are excluded from video pixels to avoid implying painted lane markings. Display boundaries are simplified within0.03m; raw map geometry is unchanged. Road elevation is a planar display proxy. Cuboid faces use approximate depth ordering, not dense mesh occlusion or a certified depth image. These limitations must accompany any video-generation use.

## Fixed captions, two information scopes

`src/config/data/semantic_conditioning/retrieval_history.md` is for past/current scene retrieval. `video_clip.md` is for offline video appearance. Both emit five bounded fields: road appearance, surroundings, lighting, occluders and distinctive details, with unknown allowed. Test scores, labels and failure causes are not inputs. A new branch's geometry overrides source-view appearance prose.

The local Qwen pilot receives four original timestamps across four cameras, each resized to maximum384px; it does not inspect all71 states. Retrieval images cover four history observations; the structured window additionally includes the -2s prelude. Full-clip appearance uses sparse -2,0,+2.5,+5s source references. Sparse photos cannot prove transient events or appearance in a novel view. The full machine geometry remains outside the text model's compact prompt.

Counts describe the30m neighborhood; textual details enumerate nearest8objects and VLM context nearest12, which can include objects outside30m. An annotation does not prove photographic visibility. The ego number is net displacement, not traveled path length. Unreviewed VLM text is stored as a draft and is excluded from released retrieval text and generation prose. `video_clip.retrieval_text` is null. Formatting success is not factual correctness.

## Execution

From the repository root:

```bash
.venv/bin/python scripts/data/export_semantic_conditions.py plan \
  --database exp/scene-catalog-v1/navtrain.sqlite --output exp/my-plan
.venv/bin/python scripts/data/export_semantic_conditions.py export \
  --database exp/scene-catalog-v1/navtrain.sqlite \
  --batch-file exp/my-plan/batch-00000.json --output exp/my-batch
CUDA_VISIBLE_DEVICES=6 .venvs/augustus/bin/python scripts/data/run_semantic_caption.py \
  --database exp/scene-catalog-v1/navtrain.sqlite --bundle exp/my-batch/TOKEN \
  --model weights/vlm/Qwen3.5-2B --output exp/my-captions
```

The exporter limits a batch to128tokens, validates cached artifacts/source/code on reuse, retains rejections and does not launch all103k jobs. The103,288token batch plan is full index coverage, not full extraction or full labeling. Portable sample exports duplicate map/window data for inspection; large production runs should store shared map assets and per-log tracks once, and use clip references. Do not pre-render every window before retrieval identifies a need.

## Limited-detail GS experiment

`experimental_scene24_v1` is a new explicit, operator-authorized profile. It retains the24dB **source-view** prefilter and permits limited texture blur. Independently reviewed, sealed full-resolution binary void masks must show total void area<=5% and largest connected region<=2%. These are chosen experiment limits, not SimScale thresholds.

Small static appearance uncertainty is allowed only for supported static classes with immutable source binding, positive Gaussian support, confirmed physical alignment, explicit uncertainty acceptance and a conservative projected area<=0.5% of the frame. Dynamic actors, bad geometry, missing frames, unsupported height, severe alignment errors, physical contacts and PDMS requirements are not waived. Old profiles and old rejected artifacts remain unchanged. This profile cannot be used by merely asserting “no holes” or by passing PSNR alone.

Text prompts and whiteboxes improve conditioning specificity; they do not establish a generator's multi-view or temporal accuracy. Generated RGB still needs the independent geometry and image checks before driving-model training.
