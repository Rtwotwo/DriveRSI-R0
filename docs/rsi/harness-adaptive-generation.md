# Contextual generation recovery

The existing continual Harness remains the single executor. After a settled family misses quota it diagnoses the actual outcome counts, chooses a registered adjustment, prepares an immutable new family, and runs normal generation/rendering/publication/training/evaluation transitions. No desktop heartbeat is required.

- Teacher rejection dominates: reduce history perturbations; PDMS must remain strictly above 0.95.
- Duplicates dominate: allow measured spatial OR sustained speed novelty, with a temporal displacement floor. Speed-only novelty is explicitly labelled.
- Other known shortages: explore untried bounded proposals.
- Unresolved execution: reconcile first; never repeat an unknown native or API task.

`adaptive_diversity.py` compiles bounded numerical criteria from source history. New profile `experimental_numeric25_adaptive_v1` requires temporal ADE >=0.20m on full and future segments, plus common-arc path ADE >=0.20m (0.30m for initial spatial preference) OR full/future speed MAE >=clip(0.04*source_speed,0.15,0.50)m/s with at least1s sustained difference. These are engineering diversity criteria, not safety certification. Source-pose PSNR remains strictly above25dB; other native publication checks are unchanged. The old profiles retain their meanings.

`generation_advisor.py` reads four current navtrain photos and measured history context using the already configured1017 private service. It can suggest a preference and image observations, not numerical safety thresholds. Three calls per campaign maximum, cached source receipts, no unknown-outcome retry. This does not generate cluster captions or update feature indices.

`adaptive_generation.py` records failure counts, actual worker time, charged attempts, action and next-round link. Two generation revisions per learning round and the existing96-started-attempt/source cumulative cap bound retries. Recovery action is selected per source; a previously unseen source starts with its initial policy. Each source retains its independent speed envelope and visual advice. Old samples/receipts are not reclassified or silently imported into new quotas.

`adaptive_handoff.py --root <campaign> --prior-pid <pid>` is a one-time version handoff: wait for prior native batch to settle, check all physical jobs including historical completed jobs, hold the curriculum lock, install the requested policy and run the existing controller. It is not a recurring monitor. A native early offroad rejection is a known rejected outcome, not an unknown crash or a training sample.

Explicit future-window termination is also rejected deterministically. `generation_outcomes.py` requires an observed terminal physical event, fewer than50 native steps, no score, reproducible execution and sealed source files. The branch provider publishes a failed receipt without inventing an RGB window. Legacy incomplete-window exceptions require matching teacher/admission receipts during reconciliation. An unknown exception remains blocked. `--resume` permits one pre-install handoff recovery only; prior launch evidence stays intact and no native/API work is retried.

## Verified external design references

[SimScale, section2.3 and appendixB.1](https://arxiv.org/html/2511.23369v3#S2.SS3) uses a human-trajectory vocabulary, bounds perturbation endpoints, sparsifies coverage, and screens feasibility before reactive rollout and rendering. Its reference envelope is longitudinal±20m, lateral±2m, heading±20°, with5m/0.5m sampling spacing. These are not appropriate to copy unchanged into our two-second perturbation controller. Our `bounded_endpoint_coverage_v2` orders the entire smaller reference grid by farthest-first coverage over endpoint and speed axes, then applies the existing finite budget. The5m/0.5m scales guide ranking only; proposed endpoints are not actual executed states or safety evidence. Final native PDMS and executed-trajectory diversity checks still decide admission. This is an adaptation, not a reproduction of their sampler or vocabulary.

The public [SimScale dataset loader](https://github.com/OpenDriveLab/SimScale/blob/07c3305c5cd4c4d517ce913749bb6f713b81e9b1/navsim/planning/training/dataset.py) cumulatively adds synthetic caches to real data. Its [DiffusionDrive training script](https://github.com/OpenDriveLab/SimScale/blob/07c3305c5cd4c4d517ce913749bb6f713b81e9b1/scripts/training/run_diffusiondrive_training_syn.sh) is a co-training recipe, not evidence that tiny pure-synthetic fine-tuning preserves generalization. The current user-requested pure-synthetic experiment remains unchanged, with full evaluation before promotion.

[R2LPL's controller](https://github.com/Engibacter/R2LPL/blob/e4b95b606aeeda35f245f130306f7150113280cd/run/script/run_rollout_cl_auto.py) carries checkpoints and replay manifests across rollout/learning rounds. DriveHarness retains its own promotion/rollback and cost accounting; it does not import that project's changed learning objective, claim their results, or replace the native student. No cross-project runtime imports were added.

Training remains pure synthetic with original ChainFlow model/loss. Each selected source must reach quota; target re-evaluation precedes global-regression checks and promotion. This implementation does not prove student gains, choose arbitrary new skills, repair arbitrary code, or guarantee every source can yield10 samples.
