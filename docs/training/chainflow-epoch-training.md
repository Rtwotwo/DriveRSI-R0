# ChainFlow stage1 epoch training through DriveHarness

The schema4 backend uses the existing lifecycle/owned-process supervisor and native ChainFlow forward, loss, optimizer and PDM scorer. No imports execute models from clones. Downloaded initialization stays in weights; all attempts, checkpoints, metadata and feedback live in exp.

## Launch and recovery
Use `scripts/training/run_chainflow_epoch.py --request PATH --output NEW_EXP_DIR --execute`. Resume adds `--phase resume --previous PREVIOUS/training-result.json`; evaluation uses `--phase evaluate` and may pass `--evaluator` with the independent scoring contract. Omit --execute for preparation only. Never rerun a settled attempt directory.

Request schema_version4 retains the existing pilot keys(seed,batch_size,evaluation_steps);batch_size is PER GPU. Model settings require `distributed_training: {devices:[6,7],checkpoint_interval:100,synthetic_every:8}`. Only devices explicitly listed are exposed to torchrun. Single-host1–8devices supported;2devices tested. Evaluation uses the first declared device and existing official navtest scorer. Manifest schema2 selects full officialnavtest;schema3 requires a sealed named subset. Synthetic samples always require current physical/visual admission. Missing inputs are errors, not silently skipped.

Global sampler shuffles the complete official original population once per epoch with seed+epoch, inserts one synthetic after7originals, and reports deterministic final-batch padding separately. All originals remain in epoch coverage. Under fixed update counts, treatment has fewer original exposures than control;the estimand is equal-compute mixing,not equal-original-exposure addition.

Checkpoint publication is atomic after all ranks reach the same update. It stores model,optimizer,scheduler,per-rank RNG and native buffers,epoch/offset/counters and exposure trace. Rank count,data,code and recipe must match on resume. A failed or budget-exhausted worker attempt retains failure status and reserved cost;after the owned process tree stops,a complete interval checkpoint may be resumed from that failure receipt. Uncommitted tail updates are repeated. Loss of the orchestrator before it publishes a terminal receipt is not verified/supportable through this entry yet.

A budget is a per-attempt cap. Resume executes min(cap,remaining native schedule). Completed horizon rejects further training. Do not alter schedule parameters to resume;start a separately versioned campaign when intentionally changing recipe.

## Paired Harness route
The existing `scripts/rsi/run_chainflow_round.py --config ROUND.json --execute` now accepts `epoch_training:true,batch_size:1`. Both arms retain complete navtrain and the same released9405parent,settings,native loss and optimizer budget. Treatment adds admitted branches. Control/resume/eval,treatment/resume/eval/replay and feedback remain existing typed stages. `compile_paired_task` checks global-batch schedule capacity before launching. A rejected export cannot start either arm.

## Numerical scope
FP32 is retained from the prior adapter;the official default is mixed16. DDP averages native gradients,uses native rank-local buffers and includes per-rank buffers in recovery. Export/evaluation uses rank0 state. This is not proof of complete published training-recipe reproduction. Small pilot schedule size128/epochs1 gives64steps/6warmup with globalbatch2. Full training must explicitly choose its native schedule;metadata population size and schedule dataset_size have different meanings.

## Evidence and remaining scale work
See exp/chainflow-ddp-rsi-v1. Full103288member coverage and CPU epoch traversal pass;real2GPU updates,checkpoint/restoration and32officialnavtest evaluation ran. Original-only4-update score fell94.0358→93.8377;no promotion. No full epoch or full12146test rerun in this change. The old synthetic branch was re-rendered and checked by the existing deterministic admission component but failed ambiguous measured road secondary layer. A qualified hard-case pair and promotion/recollection remain blocked,not claimed complete.

Checkpoint interval trades restart work for storage;this backend retains checkpoints and does not auto-delete evidence. Native per-rank scoring and synchronous loading are still throughput limits. GPU-hours,full-epoch duration and8GPU scaling need a bounded profiling run before a full campaign.
