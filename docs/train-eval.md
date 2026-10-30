# Training and evaluation

[Installation](getting-started.md) · [Run and inspect](running.md) · [Data contracts](data-contract.md) · [Harness workflow](workflow.md)

DriveRSI orchestrates native models through model-specific environments and typed requests. It does not replace a model's architecture or loss with a shared surrogate. The instructions below distinguish preparing a request from actually launching GPU work.

## 1. Prepare the native environment and model configuration

Install the selected backend using [the environment guide](getting-started.md#4-install-native-environments-separately), then create an explicit local configuration:

```bash
python scripts/build/materialize_config.py \
  --input src/config/agents/chainflow_vla/lifecycle.yaml \
  --output src/config/local/chainflow.json
```

Set the interpreter, parent checkpoint, backbone, maps, and allowed GPU resources in the resulting file. Native datasets, caches and optimization settings belong to the model's training request; a lifecycle preset alone is not a complete training recipe. Materialization never overwrites an existing file.

Inspect the model lifecycle interface:

```bash
driversi model --help
driversi model --config src/config/local/chainflow.json \
  --operation preflight --output exp/my-run/native-preflight
```

Check the resulting plan/report. Omitting `--execute` prepares the lifecycle operation rather than running its native worker. Native execution through `driversi model` requires both `--execute` and an explicit `--max-wall-seconds N`. Supply them only with configured resources and complete inputs; see the model adapter for operation-specific requirements.

## 2. Freeze the comparison

For a real-data replay experiment, declare two arms with the same parent, model/loss, seeds, total update budget, learning-rate schedule, evaluation protocol and checkpoint policy:

| Arm | Data | Purpose |
|---|---|---|
| Real-only continuation | Original training samples | Measures the effect of continued training alone |
| Real + synthetic replay | Original samples plus admitted generated samples | Measures the incremental effect of new experience |

Specify replay composition and record actual per-sample exposure. Do not equate 50 samples with 50 updates or assume one exposure is enough to learn a behavior. Pure-synthetic SFT is a separately named ablation. Keep every generated sample's source, image contract, PDMS, source PSNR and branch-binding result.

## 3. Prepare and execute a ChainFlow request

The maintained entry uses a **schema-version-4 request**. Its fields and validation are defined in [training contracts](../src/training/contracts/job.py), and the lifecycle is in [orchestration](../src/training/orchestration/lifecycle.py). Use a fully prepared request, not an old experiment file with blindly substituted paths.

```bash
python scripts/training/run_chainflow_epoch.py --help

# Preparation: validates the request and writes the native training plan.
python scripts/training/run_chainflow_epoch.py \
  --request /absolute/path/to/training-request.json \
  --output exp/my-run/real-only-preview --phase train

# Execution: same request, but a separate NEW output directory.
python scripts/training/run_chainflow_epoch.py \
  --request /absolute/path/to/training-request.json \
  --output exp/my-run/real-only --phase train --execute
```

Preparation and execution each require a new output directory; the CLI calls `prepare` on every invocation and does not accept an already-created output. The preview above is inspection only, not a resumable launch handle. Prepare a separate request/output for the mixed arm. A prepared plan is not a completed training result. Retain the parent identity, resolved data membership, complete optimizer/scheduler state and the resulting checkpoint receipt.

## 4. Resume and evaluate

```bash
python scripts/training/run_chainflow_epoch.py \
  --request /absolute/path/to/training-request.json \
  --output exp/my-run/resume-check --phase resume \
  --previous /absolute/path/to/previous-training-result.json

python scripts/training/run_chainflow_epoch.py \
  --request /absolute/path/to/training-request.json \
  --output exp/my-run/evaluation --phase evaluate \
  --previous /absolute/path/to/previous-training-result.json \
  --evaluator /absolute/path/to/evaluator-config.json
```

These prepare the requested phase. For native execution, call the appropriate command with `--execute` and a separate, new output directory (for example `exp/my-run/resume-execution` or `exp/my-run/evaluation-execution`). Do not reuse the preview directory: every invocation calls `prepare` and rejects an existing output. A recovery checkpoint includes more than model weights: optimizer state, scheduler state, progress, data ordering and RNG state matter to replay equivalence.

Evaluate the parent and both arms under the same declared NAVSIM protocol. Keep PDMS and EPDMS caches and results separate. Missing extended-comfort evidence is not a full EPDMS score. Report target repair, mean score, newly zero-scoring scenes, collision/road violations, and regressions on previously repaired scenes.

## 5. Promotion and subsequent tasks

Target improvement is a local result. A candidate replaces the retained parent only after the campaign's regression and safety requirements pass. Missing evaluation, unresolved execution, or a failed safety gate retains the parent and records the cause. Configure the next task through Harness; a checkpoint's existence alone does not authorize another training round.

## Frozen AUGUstus is a separate route

Skill experiments keep model weights and the action decoder fixed. Doctor proposals change pre-action skill text or applicability; Harness validates changes, runs cross-log and historical regressions, and promotes or rolls back the skill version. It does **not** run the SFT commands above. See [frozen skill evolution](rsi/frozen-skill-evolution.md) and [conditional skill memory](rsi/conditional-skill-memory.md).

## Upstream guides

This document uses the preparation → execution → evaluation organization of [ChainFlow-VLA](https://github.com/AFARI-Research/ChainFlow-VLA/blob/main/docs/train_eval.md) and the explicit data/cache/weight setup of [DrivoR](https://github.com/valeoai/DrivoR). Their model-specific flags are not interchangeable with DriveRSI request fields. The bundled native revision, not a newer upstream default, governs a reproducible experiment.
