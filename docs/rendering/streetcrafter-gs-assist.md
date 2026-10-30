# StreetCrafter assistance for recorded physical branches

This optional observation backend never changes physical states, trajectories, rewards or admission thresholds. The October 24 pilot uses a pinned StreetCrafter multi-camera-trained checkpoint, not the unmodified OpenDriveLab Vista architecture. It is a single-camera partial diagnostic, not qualified visual training data.

## Verified checkpoint

`weights/streetcrafter/vista_condition_multi_cameras.safetensors` contains 9,571,564,284 bytes and 2,454 tensors. SHA-256: `491c86d4868d323f719854c047d904a0fcd9fa14b918bd832c568ea7d7d51148`. The public mirror and pinned revision are recorded in `exp/streetcrafter-gs-assist-v1/download.json`; byte identity with the unavailable official Google Drive download has not been independently verified. Native strict loading has no missing or unexpected keys. The user's root-level upload is separate and untouched.

## Executable boundary

1. A sealed physical branch supplies actual ego and actor states. `src/rendering/video/streetcrafter/gs_conditions.py` reuses the native MTGS renderer for 25 real 10Hz captures. This pilot is explicitly traversal2 / CAM_F0; extending asset families and cameras requires validation.
2. The preparation receipt binds optimized camera extrinsics, intrinsics, distortion, terrain response, GS RGB, alpha mask and projected colored point clouds to the same executed tick. Nearest source captures must be unique and map to the same rounded physical tick; there is no image padding or interpolation. Actor bindings must match actual poses and presence.
3. `src/rendering/video/streetcrafter/diagnostic.py` validates sealed files before inference. Without `--gs-sequence` it uses native point-cloud-conditioned generation (50steps, CFG2.5). With that argument it uses actual GS images as latent initialization, `training_free_guidance=True`, `scale=0.3` (15 active steps of the unchanged 50-step schedule). Seed23 and1024x576 are fixed. Native model and loss are not modified.
4. The backend writes generated RGB, copied input RGB/guides/masks, branch identity, checkpoint hash, per-stage timings and failures to a fresh `exp/` directory. It always records `trainable=false`, `visual_admitted=false`, `complete_conditioned_video=false`.
5. `scripts/legacy/reports/export_streetcrafter_diagnostic.py` exports actual hashed outputs to a playable GS/point-cloud/generated comparison page.

```bash
.venvs/streetcrafter/bin/python src/rendering/video/streetcrafter/diagnostic.py \
  --prepared exp/conditioned-video-loop-v1/streetcrafter-conditions-v2/report.json \
  --checkpoint weights/streetcrafter/vista_condition_multi_cameras.safetensors \
  --expected-checkpoint-sha256 491c86d4868d323f719854c047d904a0fcd9fa14b918bd832c568ea7d7d51148 \
  --gs-sequence exp/streetcrafter-gs-assist-v1/gs-conditions-v3/report.json \
  --output exp/streetcrafter-gs-assist-v1/refinement-new-run \
  --camera CAM_F0 --max-clips 1 --max-wall-seconds 1800
```

Select an available GPU explicitly. Do not run this example over an existing output directory. Refer to the saved launch receipt for environment settings. Source files are part of sealed input provenance; changed code must not be silently substituted for the recorded version.

## What is not established

- Multi-camera-trained weights do not imply joint multiview consistency. This test covers only CAM_F0, physical ticks0..24 (2.4seconds), not the full four-history/ten-future window.
- Native mask blending is disabled upstream. Supplying masks or accurate point clouds does not force the diffusion output to preserve vehicles, road boundaries or occlusion.
- The GS input itself has ambiguous measured road layers for ego ticks0..5. That diagnostic fallback is preserved, not certified. Other ticks passing terrain lookup does not establish independent 3D registration.
- Optimized GS optics differ from raw photographs. Output-versus-GS measurements report image change, not quality. The point-cloud-only run's first-frame reconstruction uses its conditioning image and is not held-out PSNR.
- Offline reconstruction and bidirectional video conditioning have not passed a history-causality test. Future conditions must not be used to assert an online observation is causal.
- Generated outputs need object/geometry, temporal, multiview and held-out quality checks before any training admission. This turn performs no training and makes no learning-benefit claim.

Evidence and actual timings: `exp/streetcrafter-gs-assist-v1/implementation-ledger.json`.
