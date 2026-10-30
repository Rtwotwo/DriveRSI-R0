# Dynamic assets, history images and native replay comparison

The maintained entrypoint is `driversi.rsi.experiments.dynamic_history_campaign`.
It registers `dynamic_history_replay_campaign` in the existing Harness engine;
the final training stages are compiled by `compile_paired_task`, not by a new
optimizer or trainer. This campaign is a bounded experiment, not proof that
data augmentation improves the model.

## Automatic sequence

1. Freeze capture recovery jobs, physical generation receipts, diversity
   receipts, the original-data manifest and the native model recipe.
2. Reconstruct actual background and supported dynamic foreground assets with
   the existing native MTGS implementation, on at most two allocated devices.
   Unsupported actors remain explicit. A `completed` asset is not yet a
   source-view PSNR or novel-view admission.
3. Reuse the selected immutable physical rollouts, without new teacher calls.
   Regenerate branch images and native packages using the actual dynamic bundle.
   Bind the ego, actors, times, calibration and image state to that one branch.
4. For the current ChainFlow Stage1 consumed-input contract, publish only the
   current anchor × four cameras (4 RGB images). All four historical and ten
   executed future numerical frames and native trajectory targets remain;
   prior nonanchor RGB and future RGB are absent. Native verification rejects
   treating an unrendered frame as a new observation anchor. The earlier
   16-image history contract remains a separate readable format, not relabelled.
5. Admit only strict PDMS > 0.95, mean source-view PSNR > 25 dB and the dynamic
   binding profile `dynamic_anchor25_v1`. These are not novel-view pixel truth.
6. When the bounded cohort meets its sample and distinct-source floors, compile
   and execute the ordinary real-only control and real-plus-synthetic treatment.
   Both use the same ChainFlow Stage1 parent, seed, native model, loss, optimizer
   schedule and update budget. The existing mixed sampler inserts one synthetic
   sample per eight global samples; this is a controlled baseline, not a claimed
   optimal SimScale ratio.
7. Verify positive-learning-rate exposure and exact cross-process recovery.
   Test the original failed query first. A still-zero result stops full-test
   cost and retains the parent. Otherwise run both full navtest evaluations,
   compare against the same parent, and apply the existing release decision.
   A campaign-local incumbent can advance only after target improvement,
   overall non-decrease, no safety regression and no new zero. No global
   deployment pointer is modified.

## Bounds and accounting

The initial plan selects two sources and six existing branches per source:
at most 12 new image/log packages. At least four admitted packages and two
contributing sources are required; one source cannot fill the other source's
coverage. Actual admitted count determines training budget. With four requested
synthetic exposures, global batch four and the 1:7 mix, 12 accepted packages
give 96 optimizer updates per arm (48 synthetic + 336 real exposures in the
treatment); four accepted packages give 32 updates. The independent resume
verification repeats half the treatment updates and is recorded separately.
The schedule horizon is declared once for the bounded maximum; receipts record
actual learning rates and actual sample exposures.

`progress.json` links each branch to its source, result and native package.
`training-input.json`, the compiler's `intervention.json`, the native task
receipts and the release decision record the model, budget, real/synthetic
exposure, recovery, evaluations and decision. Wall time includes asset work,
rejections and waiting. Reused physical rollouts are explicitly counted as
reused; they are not reported as newly generated teacher rollouts.

## Recovery semantics

Each source and branch has a `started.json` marker and a durable final receipt.
A restart consumes completed receipts. If an adapter saved its final
`execution/report.json` but the controller was interrupted before settlement,
the controller reconciles that report without repeating reconstruction or
rendering. A known unsupported asset or explicit rejection advances to the
next declared source/branch. Native reconstruction handles its own bounded,
checkpoint-aware recovery. Exceptions or missing final results are unknown,
not low-quality trajectories; no GPU work is blindly reissued. Unknown work
blocks training until reconciled. Completed training uses the existing native
Harness ledger and is never repeated merely because the outer controller was
restarted. The batch is finite and does not create a recurring automation.

Existing dataset packages are preserved. Runtime activation selects the
`sensor_blobs/` and `navsim_logs/` consumed-image layout and a training-only
publication policy; feature extraction and caption calls remain deferred.
The previous publication policy is recorded before the campaign binding
changes. No files are cleaned up by this entrypoint.

## Versioned consumed images and existing asset adoption

`history_rgb_future_states_v1` uses admission schema 3, 16 history RGB images
and `dynamic_history25_v1`. `anchor_rgb_history_future_states_v1` uses schema 4,
4 anchor RGB images and `dynamic_anchor25_v1`. The campaign validates the exact
combination; old 16-image receipts cannot fill an anchor-only quota. The new
campaign has its own task ID and dataset collection activation.

A source may point to a `prebuilt_asset_job`. If its native job is still
executing, the new campaign may adopt it only with the explicitly recorded
host, PID and process start identity. The exact job command and process group
must match. It waits within the asset deadline, records `adoption.json` and
does not launch a duplicate or terminate the adopted job. Completed output
digests are checked before reuse. If the original process group has fully
exited without a final receipt, the existing native job can resume its saved
checkpoint once through the bounded stage. An unknown live group blocks the
handoff. Per-device locks prevent a prebuilt job's saved GPU from colliding
with another source's allocated lane.
