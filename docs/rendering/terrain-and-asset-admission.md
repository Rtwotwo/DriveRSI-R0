# Terrain and asset admission

Height means the road/object reconstruction Z coordinate, not a software layer. Two supported local surface fits are evidence of ambiguous geometric support, not proof of a physical overpass. Search exhaustion is separately reported as unresolved. `terrain_failure_evidence` preserves the complete original rejection evidence and never changes support, tolerances, or admission.

## Asset inventory and route representation

Run `python scripts/rendering/catalog_worldengine_assets.py --output exp/<run>/assets.json`. The catalog follows every bounded metadata page, checks stable counts, unique archive paths and versioned nonempty entries, and rejects partial inventories. On this run, the former first-page list of 100 was expanded to all 129 archives. This is metadata coverage, not qualified scene coverage. Downloads remain an explicitly bounded separate operation.

Task routes may contain up to 8,192 points. The observed 4,143-point route is preserved exactly, without reducing resolution or changing endpoint/traffic/scoring rules. Larger routes still fail explicitly.

## Actual bounded experiment

Evidence is under `exp/terrain-paired-rsi-v1/implementation-ledger.json`. Twenty-seven downloaded bundles from eight navtrain source logs were screened. Missing dynamic nodes are only a prefilter signal: an immutable static object may legitimately be in the background, but needs appearance/position evidence. Do not drop it from physics to satisfy a renderer.

Two native DriveRL arrival/continuation branches completed with exact replay and no contact. Pittsburgh scored 98.4841 and Singapore 100 using native four-second counterfactual PDM mathematics, with five actual seconds of future execution. Their respective native DriveRL baselines have the same scores. Neither branch has established a ChainFlow weakness or teacher improvement; neither is learned self-play.

Three complete four-camera renders generated 180 novel-view images and 180 source-pose references. The two primary assets have finite terrain support. The alternative Singapore asset does not have complete terrain support. Source-pose PSNR is 28.25 / 22.20 / 22.12 dB, respectively. These exposure-normalized source-pose numbers are not held-out novel-view quality scores.

Pittsburgh remains rejected because small static appearances are unresolved. Singapore's primary asset has image-quality and object-binding failures; the alternative also has terrain failure. Two real Harness export attempts produced explicit visual rejection receipts and no training. The original MTGS height ambiguity also remains rejected. No native model, loss, scoring rule, or admission threshold changed.

## Downstream boundary

The ChainFlow round accepts only admitted synthetic export receipts. It must not turn a renderer's diagnostic completion into a training sample. Keep the 9405 parent until actual paired training and locked navtest evaluation produce an acceptable benefit report. This experiment generated no new training, navtest score, promotion, or recollection result. Earlier two-GPU backend/recovery tests are separate evidence.

The source member token, source log and reached physical anchor are separate fields. Pittsburgh raw89 is an official member while its reached raw93 anchor is derived; Singapore raw131 and raw135 are official members. Derived branches inherit their verified training source, not a fabricated benchmark identity.

Local demo: `outputs/demonstrations/terrain-paired-rsi-v1/`. Release staging: `<repository>/terrain-paired-rsi-v1/`. Nothing is pushed by this export.


## Portable render terrain repair

`python -m driversi.rendering.diagnostics.terrain_repair --render-report <report.json> --output <new-directory>` derives a repair from the exact physical motions, asset coordinates and source poses recorded by any complete portable render. It checks every source and target query, repairs only NaN cells using the existing bounded original-surface fit and spatial holdouts, and preserves all measured cells. Reopening the same output validates and reuses the saved repair. No scenario IDs or manual pixel coordinates are encoded in the implementation.

New reactive postprocess workers automatically dispatch this bounded CPU repair when terrain support is missing. The original render remains rejected. `ready_for_rerender` means the derived grid can be used in a separate same-branch render; it does not grant visual or training admission. `additional_measured_support_required` means the local repair cannot infer the surface and more reconstruction support is needed. The current active worker is not hot-patched; its already completed rejections are preserved.

Whitebox planar display is not a metric GS terrain source. Do not flatten the reconstructed road, overwrite actor height or reuse a source-view PSNR number as a new-view geometry certificate. Latest production policy remains strictly greater than 25 dB; full physical, appearance, occlusion and temporal admission still apply.
