# DriveHarness workflow

Two research routes share the execution and evidence layer. They have different changing state and different success criteria.

| Route | Changes | Fixed for a controlled comparison |
|---|---|---|
| Experience acquisition and native learning | Source selection, bounded generation policy, admitted data and candidate model weights | Scoring protocol, admission rules, native model/loss, parent identity and paired update budget |
| Frozen-model skill repair | Skill text, applicability, bounded calls and persistent experience | VLA weights, action decoder, evaluation and safety gates |

## Acquisition → rendering → learning

1. **Record feedback.** Freeze the evaluated parent and protocol. Select explicit development failures. Reusing navtest feedback is an adaptation experiment; use separate logs for an unexposed evaluation claim.
2. **Retrieve training sources.** Use the training catalog and selected visual/text/state method. Caption and cluster membership are evidence for retrieval, not proof of a common failure cause. Do not use test scenes as training sources.
3. **Plan bounded generation.** Declare source IDs, available assets, teacher runtime, attempts, timeout, GPU allocation and desired cohort size. Adaptive diversity can vary with scene context; safety qualification cannot be relaxed to fill a quota.
4. **Execute the physical branch.** Generate actual perturbed history and a corrective continuation. Retain all object states, identities, timestamps, termination events, teacher scores and failed attempts.
5. **Qualify trajectories.** Check PDMS strictly greater than 0.95 for the current profile, completed execution, geometry and the declared difference rule. Spatial-path and speed diversity are reported separately.
6. **Prepare and render assets.** Bind real reconstruction metadata to the source. Render camera poses and participating objects from that same physical branch. Group rendering by asset where supported. Use the student's declared RGB horizon, not a hard-coded future image schedule.
7. **Admit and export.** Check source-view PSNR, image/camera/object identity, synchronization and native loader compatibility. Only admitted samples enter `dataset/`. Uncertain or absent object appearance is a rejection or reconciliation issue, not permission to substitute an unrelated object.
8. **Run the native paired comparison.** Fix the parent, seed, optimizer/scheduler recipe, updates and evaluation. Compare real-only continuation with real-plus-synthetic replay. Record actual exposure of each synthetic sample. A pure-synthetic route is a separate ablation.
9. **Evaluate and decide.** Target improvement and global preservation are different checks. Evaluate the target, prior repaired problems, and the declared regression/full benchmark. Retain the parent on failed or incomplete evidence. Promotion is campaign-local until the required evaluation is complete.
10. **Write experience.** Preserve conditions, actions, failures, costs and decisions for later acquisition plans. Memory being read is not itself evidence of learning benefit.

## Main entry points

```bash
# List concrete services and their contracts.
driversi services
python scripts/rsi/run_platform_spine.py inventory --output exp/example/interfaces.json

# Inspect the existing generation and continuation interfaces.
python scripts/rsi/run_synthesis_campaign.py --help
python scripts/rsi/run_platform_spine.py continue --help
python -m driversi.rsi.experiments.dynamic_history_campaign --help
```

The dynamic-history campaign accepts an explicit `--plan` and `--output`. Its schema is defined by `validate_plan` in [the implementation](../src/rsi/experiments/dynamic_history_campaign.py). It requires 2–3 declared training sources, sealed generation/diversity reports, real reconstruction jobs, bounded resources and a prepared native replay manifest. It reuses physical rollouts; it does not create a missing teacher rollout for you. **Invoking this campaign runs work**; `--help` is read-only.

The latest dynamic campaign is experimental and currently paused; its real-data limitations are listed in [release status](release-status.md). No runnable private experiment plan is shipped, because such plans contain source-specific evidence and machine paths.

## Persistent task control

```bash
# First inspect the complete CLI.
python -m driversi.rsi.workflow.engine --help

# Commands for an explicitly registered task:
python -m driversi.rsi.workflow.engine --registry exp/my-run/harness.sqlite status TASK_ID
python -m driversi.rsi.workflow.engine --registry exp/my-run/harness.sqlite events TASK_ID
python -m driversi.rsi.workflow.engine --registry exp/my-run/harness.sqlite cancel TASK_ID
python -m driversi.rsi.workflow.engine --registry exp/my-run/harness.sqlite recover TASK_ID
```

`cancel` requests termination of owned active work. `recover` reconciles a dead owner and recorded children; **it is not an unconditional rerun**. Finished tasks cannot be claimed again. Known checkpoint or storage recovery is bounded and must preserve identity and spent budget. Unknown native outcomes stop dependent stages until reconciled. Explicit user pauses take precedence over autonomous progression.

Task DAGs, native execution receipts and campaign summaries retain different levels of evidence. Read the final report and process ownership, not just a stale progress file or PID number. Do not delete a failed run's receipts to make a retry look fresh.

## Frozen-model skill route

The implemented route is failure evidence → Doctor proposal → modification/duplicate checks → frozen AUGUstus planning → cross-log and historical safety regression → promotion or rollback → persistent experience. Text-only, applicability-only and combined modifications can be compared separately.

See [frozen skill evolution](rsi/frozen-skill-evolution.md), [conditional skill memory](rsi/conditional-skill-memory.md), and the source index in [code map](code-map.md). Model weights are unchanged on this route; generated plans and feedback are evaluation evidence, not SFT samples. It requires a configured Doctor service if automatic proposals are desired. The existing trials establish execution and rollback behavior, **not sustained driving improvement**.

## Reproducibility record

For every new run keep: source revision; parent/checkpoint identity; split and source groups; model and image contract; runtime configuration; acquisition and sample manifests; all acceptance/rejection reasons; attempted/accepted counts; summed stage times and wall time; data exposure and training updates; target/global metrics; promotion or rollback decision.

Never compare task-time sums to wall time without accounting for parallelism. Never turn missing EC into a claim of complete EPDMS. Never count a rendered frame or a completed asset as an admitted training sample.
