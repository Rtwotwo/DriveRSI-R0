# Harness asset recovery

A missing or underquality source asset enters a finite recovery service. A source report must contain all60 measured four-camera comparisons and bind both image files, metadata and checkpoint. Its mean must be strictly greater than27dB. This source preflight does not grant synthetic training admission.

The owned modules are `driversi.rsi.experiments.asset_recovery`, `driversi.rsi.services.reconstruction`, and `driversi.rendering.gaussian.mtgs_reconstruction`. Existing assets can use appearance refinement, bounded background geometry refinement, then measured-log reconstruction. Missing assets begin with reconstruction. Calibration and actor identities remain controlled; new branches still pass native trajectory, physical, visual and supervision checks.

A campaign registers `asset-recovery-policy.json`: strict_source_psnr_gt=27, selection=available_within_frozen_visual_pool, controller interpreter, free builder GPUs, fixed existing source reports, and a source preparation factory. This controls readiness only and never imports geographic selection or new test-derived training samples. Pending sources remain recorded; available sources each retain their own10-sample minimum.

The automatic executor reconciles background recovery receipts before selecting a problem. It waits for resources within its wall budget, requeues recovered sources, and distinguishes known exhaustion from unknown execution. An unknown in-flight task is never repeated automatically. Each strategy has a fixed request, immutable receipts, cancellation of its own process tree and a bounded attempt/time allocation. A failed alternative is evidence, not an admission override.

Raw reconstruction uses owned MTGS and nuPlan code: source window, semantic masks, measured LiDAR registration, COLMAP fixed-pair triangulation, depth, colored measured points, native reconstruction training, export and measured road support. No source-frame interpolation or invented LiDAR is used. The isolated runtime has a seeded GS CUDA forward/backward and real-log preprocessing witness, verified again in a fresh process before release. Dependency versions and model hashes are recorded. The COLMAP4.0.1 pair API preserves the prepared image-pair list; bitwise parity with3.11 is not claimed.

A declared `generation-recovery-policy.json` can enable one alternative after confirmed duplicate-limited sampling: bounded historical perturbation followed by native DriveRL correction. It is explicitly a composite generator. It retains the same source set, model/loss, quality profile, per-source quota and independent diversity checks. Other failures remain typed and unresolved rather than silently retried.

`execution-placement.json` records an authorized host move without changing sealed earlier rounds. On synthetic data completion, existing curriculum training/target retest/full-regression/release gates remain authoritative; source recovery cannot promote a student.

Current bounded command: `.venv/bin/python -m driversi.rsi.experiments.continual_curriculum auto --output exp/continual-family-pdms-v1/adaptive-problem-27db-v1 --max-problems 3 --max-transitions 18`. Only one runner owns the campaign lock. This is an experiment process, not a recurring desktop monitor.
