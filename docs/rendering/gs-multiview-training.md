# Four-camera GS training pilot

The raw 3DGS path is the selected observation generator for this pilot. The native StreetCrafter joint temporal recipe changes generated history when only later GS/guide conditions change. An identical-input repeat is deterministic. This is a sensitivity probe, not a valid synthesized driving scene or a universal conclusion about every video model. Its images are diagnostic-only.

The same sealed navtrain branch contains 71 physical states at 10 Hz. Render ticks 0,5,...,70 for four cameras. Tick 0 is additional visual evidence; history is 5,10,15,20, and future is 25,...,70. The native ChainFlow stage1 camera stack order stays F0,B0,L0,R0 and its current four-camera image input is unchanged. Earlier history and future images are archived; future images never become the current observation.

Measured road support is validated separately from appearance quality. Do not choose a vertical layer merely because two other traversals support it. A coherent alternative measured road layer must remain a rejection. Existing fit tolerances and the explicit experimental_source24_v1 appearance cohort remain fixed. Source-pose PSNR is not novel-view ground truth.

The bounded paired experiment uses the same 9405 parent and optimizer-update budget. The base sampling population is all 103,288 official navtrain scenes; eight optimizer updates do not constitute a full epoch. Test reporting uses 32 preselected official navtest samples, one per original log, chosen without inspecting scores. It is a named partial pilot protocol, never the full 12,146-scene benchmark. Training and test membership checks remain separate. No automatic promotion follows this small experiment.

Artifacts: exp/gs-multiview-training-v2/. Downloaded initial weights stay under weights/. All new model/optimizer checkpoints, evaluation and feedback stay under exp/.

## Current execution limit

The completed Singapore render is diagnostic-only: the historical validation-v4 check supported 47/71 high-rate ego queries; this predates the residual-proof fix and is not current qualification and fails required late image samples. Earlier reports with more passes were superseded after rejection-regression fixes. No training receipt may reuse those flags. Raw single-scan measurements explain why the first six relative displacements can be stable despite absolute-plane ambiguity, but do not certify the full window.

The inspected Pittsburgh asset has explicit pedestrian/bicycle training masks and no pedestrian Gaussian nodes. Visible unbound people are an appearance-coverage failure, not an acceptable distant-object exception. No paired optimizer update has started for this run.

The local scan audit found 168 other scans from the inspected Pittsburgh log, but none of the 48 exact scans referenced by the visible pedestrian metadata in the checked roots. Additional object point counts remain unknown. Do not use unmatched nearby scans, duplicate points to satisfy initialization, relabel pedestrians as vehicles, or claim all machine data has been searched.


## Explicit static background geometry

The portable renderer now records immutable source3D bindings for native static categories and vehicles already proven parked by the complete source motion evidence. Planar simulator Z is not a full3D reconstruction height: background OBBs use their real source annotation and fixed refined source pose, bound to exact float32 physical identity. Every background binding is recomputed from sealed scene, executed states, original log and asset metadata during admission. A changed physical pose invalidates it. Geometry point counts do not grant visual admission.

An object visible in initial images may receive its first source annotation later. The explicit `survey_prior_static_native_v1` option separates physical scene existence from annotation availability for known immutable native static classes only. It requires reviewed initial-source imagery, original3D survey evidence, a sealed source identity, stationary execution and no initial overlap. It never backfills future dynamic vehicles or changes raw observation timestamps. This is an offline static world prior, not an online sensor observation.

The retained July16 v1/v2 branches document micro-velocity drift and the late static annotation mismatch; they are not silently relabelled. A new physical execution, recovery check, GS output and admission are required after applying the prior.

## Actual July16 audit

Fresh v3 execution scored99.8366 under native4s counterfactual PDM and retained5s of actual future. The final code recheck still supports all15 camera timestamps through the asset native finite RoadHeightMap (vertical support only, not a complete surface-orientation certificate).60 novel/source pairs were rendered at960x540. Mean source-pose PSNR26.4207dB (minimum22.2632) passes the existing mean24dB experimental prefilter, not a heldout27dB criterion. Independent pixel review still rejects the cone identity in9side-camera frames. Native1920x1080 rerasterization of8diagnostic views did not resolve it. The actual typed Harness export returned `chainflow_synthetic_visual_rejection`; no sample or optimizer update was produced from this branch.

## Residual surface safety correction

The legacy pooled-traversal estimator could accept a coherent second layer after discarding less than5percent of points. Independent-traversal agreement does not resolve that ambiguity. `measured_surface_residual_proof_v2` now checks exact residuals in every supported source/reference layer and in both the pooled and consensus paths, retaining all original numerical thresholds. The render admission validator requires and recomputes this proof; stale producer flags are rejected. A failed bounded secondary-plane search is not evidence that no secondary plane exists.

## Final bounded repair result

The existing6 cone Gaussians received200 actual optimizer updates to SH color and opacity only; all geometry and all other rows remained bitwise unchanged. This used native MTGS RGB loss terms, not the complete original MTGS reconstruction objective. The fixed5time holdout excluded every camera; only4of those times had a target projection, yielding6crop comparisons. Mean localPSNR change was+0.00271dB, and independent object/occlusion review still rejected the target. The holdout is specific to this refinement; original asset exposure is unknown. Native1920 rerendering had also failed.

25real measured points occupy the upper part of the annotated box and project near vehicle/vegetation boundaries; they do not independently establish cone identity or the missing lower geometry.6points come from the refinement holdout timestamp and were not used for initialization. No invented or duplicated points were added. The original asset remains intact.

ChainFlow optimizer updates remain0. The preselected32official-navtest parent baseline completed at94.035788, with0runtime failures and1driving-zero sample. This is a partial adaptive pilot, not the full benchmark or model-benefit evidence. The real rejected dataset receipt was passed to the paired compiler and rejected before task creation. No model promotion or recollection occurred.

Final related regression batch:168passed/1skipped. Other reported52/37/13test batches overlap and must not be added. Full core suite was not rerun. All source photos remain private; the demo exposes synthesized GS comparisons and scalar aggregate results only.


### Explicit source25 experiment (Singapore paired pilot)

`experimental_scene25_v1` is operator-selected, not the default `source27_v1`.
It requires mean canonical-source PSNR >=25 dB, a sealed exact-pixel review of
every frame and camera, sealed binary void masks, and explicit road, lane,
on-road vehicle and occlusion checks. Physics, native PDMS, temporal coverage,
asset identity and terrain checks remain unchanged. It is not held-out NVS PSNR.

The measured off-road pedestrian's fine texture/anatomy may be deferred in this
specific reviewed branch, while its occupancy and occlusion remain present.
Supporting roadside measurements are provenance, not a generic automatic
pedestrian exemption. Existing tiny static appearance uncertainty is separately
labelled; it still requires immobility, physical GS support, reviewed alignment
and a bounded projected footprint. Missing cars/pedestrians are never allowed.

The pilot uses the same 9405 parent and update budget for both arms, full navtrain
as the source pool, and the existing fixed32 official navtest scenes for a cheap
paired evaluation. This is neither full-epoch training nor full navtest scoring.


### Explicit ego-camera25 experimental profile

`experimental_egoview25_v1` is separately selected at the operator's request.
It keeps matched-source mean PSNR strictly >25dB, full window/file integrity,
all eight global camera checks and existing severe-void bounds. Terrain support
and exhaustive object binding failures remain diagnostic warnings. The reviewer
must inspect actual target pixels; unknown object height produces no guessed box.
Source photos depict the logged pose and traffic, not counterfactual ground truth:
expected vehicle movement relative to a source photo alone is not a defect.
Actual ghosts, missing on-road vehicles, holes, distortion or inconsistent motion
still reject. Unknown/global missing evidence is never converted to a pass.

The old source27/scene25 profiles remain unchanged. Profile-bound review evidence
cannot be silently reused under another policy. Native AUGUstus joint supervision
still independently requires trustworthy object/language labels; relaxing the
render metadata prefilter does not authorize fabricated training supervision.
Request quota exhaustion and invalid critic responses are technical failures,
not evidence of low PSNR or bad geometry. Both are retained with real raw receipts.
