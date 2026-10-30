# Frozen-policy skills through DriveHarness

This pilot tests whether a pre-action skill prefix changes frozen AUGUstus planning usefully. It does not train the student, modify native actions, or claim recursive skill improvement.

## Contract

- Student: the complete original AUGUstus SFT checkpoint, strict weight restore and unchanged native action sampling/normalization.
- Inputs: real four-frame historical images, the original native state/navigation prompt, and one conditional skill. No future actions, full-clip captions, metric cache, or prior PDMS enters inference.
- Routing: current navigation and speed select a turn, low-speed progress, or following checklist. A selected checklist is not a verified failure diagnosis.
- Three arms: unmodified native prompt; matching checklist; exactly tokenizer-length-matched neutral text. Pair seeds 42 and 43 within each observation; do not interpret seed repetitions as independent logs.
- Population: 32 distinct logs, selected from this exact parent's saved results (12 zero, 12 nonzero below 0.6, 8 others). This is retrospective navtest adaptation/development evidence, not untouched benchmark generalization.
- Before the main pilot, verify exact same-seed raw action replay and equality with/without post-action explanation. Omit explanation only if this real-input witness succeeds.
- Seal all 192 predictions before the unchanged native PDM scorer receives them. Do not select a trajectory based on its eventual ground-truth score.
- Persist changes, gains, newly introduced zeros, safety/comfort regressions, token counts, time and model identity. No automatic skill promotion based on this small pilot.

## Existing Harness services

1. `frozen_skill_observations`: validate original receipts and build the selected native observation/cache bundle.
2. `native_model_frozen_skill_inference`: one frozen model instance, paired prompts and seeds, no scorer or optimizer.
3. `frozen_skill_native_scoring`: score the sealed predictions independently, charging the native simulator budget before scoring.
4. `frozen_skill_paired_analysis`: complete-coverage analysis and candidate experience record; native policy stays default.

The task uses the existing durable workflow engine and typed artifacts. It does not use a new background monitor or an ad-hoc per-stage launcher. Completed stage receipts are retained; interrupted/unknown execution is not blindly restarted.

## Reproduction

On the project host, create an experiment config containing `task_id`, `baseline_report`, `baseline_plan`, `checkpoint`, `skills`, `count: 32`, `seeds: [42, 43]`, and `gpu`. Use an idle authorized GPU and the existing AUGUstus environment. The parent checkpoint must match the historical baseline exactly.

```sh
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
python -m driversi.rsi.experiments.frozen_skill_probe exp/<study>/config.json exp/<study>/declaration
python -m driversi.rsi.workflow.engine --registry exp/<study>/harness.sqlite register exp/<study>/declaration/task.json --output-root exp/<study>/runs
python -m driversi.rsi.workflow.engine --registry exp/<study>/harness.sqlite run <task_id>
```

Use `status` or `events` on the same registry to inspect the automatic transitions. Never register the same execution as a new task simply to replay completed inference. A failed preparation can be corrected with a versioned declaration before any model output has been produced; retain the failed receipt.

`CUBLAS_WORKSPACE_CONFIG` must be set before CUDA initialization for the declared deterministic replay. A missing setting causes the kernel to refuse execution; do not resolve that error by disabling deterministic checks. A pilot failure before any prediction may be reconciled and rerun under a new explicit runtime identity while keeping the frozen plan unchanged.

## Interpretation and next step

Changing text after action generation cannot affect that action. Pre-action skill conditioning can affect it, but improvement must be measured against native and neutral-text controls. A larger action difference is not automatically better.

This first version contains fixed human-designed checklists. A later recursive version needs a feedback-driven proposal operator, independent cross-log validation, versioned promotion/rollback, and comparisons against equal-budget nonpersistent diagnosis. Keep the current trial separate from that unproven claim.

Real recorded images support evaluation at recorded anchors. They cannot be reused as if they were new visual observations after the simulated vehicle leaves the recorded pose.
