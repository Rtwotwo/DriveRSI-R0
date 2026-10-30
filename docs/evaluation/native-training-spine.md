# Complete teacher metrics and a real-data training spine

The reusable entry is `python -m driversi.rsi.experiments.training_spine`.
It consumes a settled rendered generation, its exact capability plan, a full
official NAVSIM manifest and a native ChainFlow model configuration. It refuses
family plans: a one-sample engineering check must not bypass the ten-member
family gate or claim family completion.

## Generation and scoring

Native DriveRL acts for all 70 physical decisions. Native TTS and native
stochastic sampling are distinct declared skills. Sampling uses the upstream
distribution without changing weights, action limits or model/loss code. Its
snapshots also own Python, NumPy and Torch RNG. Sampling cannot silently combine
with TTS or continuous parity. Saved-state replay and input-parity checks preserve
the ambient random stream, including reference-model construction.

An action-only policy has no genuine four-second trajectory head. The planning
adapter therefore forecasts two independently recomputed four-second continuations
from actual worlds 0.5 seconds apart. It restores the executing world exactly.
These are privileged world forecasts, not dummy trajectories, later execution
slices, or official logged navtest submissions. Deterministic continuation can
give EC=1; this does not establish robustness to world-model error or perturbations.
Native NAVSIM metric arithmetic is unchanged. Missing EC is rejected; all five
weighted terms remain in the denominator. The strict synthetic threshold is
complete EPDMS>0.95; PDMS>0.95 remains separately required. The official score
horizon is four seconds within a seven-second executed branch.

## Asset selection and reconstruction

`rsi.decisions.asset_supported_selection` establishes a global visual Top20
before checking asset availability, continuity and source groups. It never takes
map location, absolute heading or future targets as inputs. Keep this availability
fallback distinct from earlier group-deduplicated retrieval results.

`rendering.gaussian.source_preflight` checks all four source cameras at 15 actual
captures, selected by timestamp (within50ms), never by assuming a metadata row rate.
`source_refinement` produces immutable appearance-only child assets with the
native reconstruction objective; geometry, cameras and actors are unchanged.
`source_repair_campaign` bounds optimizer work and supervises its worker with a
hard wall timeout. Views excluded from gradients still enter mean-based selection;
they are not an independent held-out test. Source PSNR does not establish novel-view
quality. Asset preflight never grants training admission.

## Singleton training-spine contract

1. Bind a fresh one-assignment plan and the selected training source.
2. Execute and render a new branch with complete teacher scores.
3. Run the existing `experimental_numeric25_v1` checks and canonical exporter.
   This operator-authorized profile requires mean source PSNR>25dB, but does not
   claim independent visual review or verified novel-view fidelity.
4. Extract fresh features from the four generated history frames. Publish the
   admitted SimScale-style sample to `dataset`, separate from original navtrain.
5. Compile the existing native paired task: same9405 parent, eight updates per
   arm, two GPUs, native FP32/loss, full original replay pool, one synthetic example
   every eight global samples in the treatment. Confirm actual sample consumption.
6. Keep resume/replay checks and full12,146-scene official navtest evaluation.
   This is adaptive evaluation. No navtrain student scoring is introduced.

For a singleton, pairwise ADE has no comparisons; it is not evidence of diversity.
Family minimums remain intact. The controller records `family_completed=false`
and `promoted=false`; it cannot publish a student as improved solely because
optimization started or one example passed.
