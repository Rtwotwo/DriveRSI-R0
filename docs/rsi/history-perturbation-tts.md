# Executed history perturbation + native TTS

This independent generator complements all-window DriveRL. It is inspired by SimScale's separation of perturbed arrival and expert correction, not a reproduction of its vocabulary/EPDMS protocol.

## Control and information contract

- Skill `perturbed_history_tts`; family `history_endpoint_grid_tts_v1`.
- At t=0 derive a source-relative target at t=2s in the rear-axle frame. Quintic XY reference matches actual initial position, velocity and acceleration, target position/velocity, and zero target acceleration. LQR lookahead continues at terminal velocity.
- The first 20 actual 0.1s controls use native LQR and bounded native bicycle dynamics. Reference positions never overwrite physical state. Privileged source motion is teacher-only.
- At the actual t=2s anchor, preserve physical world, other agents, ego steering/dynamics, time and executed history. Native DriveRL TTS (8 candidates; available native search, not gradient TTO) controls the next50 steps.
- Actual71 states/70 actions define complete7s. History image ticks5/10/15/20; future25..70 every5 ticks. Four cameras. Original model, loss and native scorer unchanged.
- Saved reference is separate from actual rollout. Actual waypoint tracking error is diagnostic; proposed offsets are not achieved novelty or training evidence.

## Independent admission

`experimental_numeric27_ade05_v1`: native PDMS strictly >95 over the anchor's first4s, source-view mean PSNR >=27dB, full7s AND future5s pairwise ADE >=0.5m. Complete actual execution, hashes, camera/ego motion and image ownership remain mandatory. Terrain/VLM/object appearance remain diagnostics under the authorized numeric experiment. This does not certify27dB held-out novel-view fidelity, nor collision-free first2s from the future PDMS alone.

Historical24dB/.3m profiles/receipts stay unchanged. No label-only fallback. A failed/incomplete path is retained and cannot become a full7s sample.

## Deterministic proposal family

Default24 proposals: endpoint (longitudinal,lateral) pairs (0,-1),(0,+1),(-2,0),(+2,0),(-2,-1),(-2,+1),(+2,-1),(+2,+1) metres, crossed with terminal speed deltas0,-1,+1m/s. Scene endpoint heading defines forward/left. The validation envelope is5m longitudinal,2m lateral,2m/s delta; this is an implementation limit, not SimScale's20m perturbation interval.

An explicit `proposal_grid` may supply10..64 distinct references within the same bounded envelope; it is frozen with the plan. Target10 per source is a hard training-start requirement, not a promise of yield. Only measured actual ADE certifies distinctness. Sideways perturbations can leave the supported/drivable road; complete-7s requirement is not relaxed to count them.

## Automatic interface

Set `generation_skill: perturbed_history_tts`, `trajectory_family.method: history_endpoint_grid_tts_v1`, `attempts_per_anchor: 24`, `minimum_ade_m: 0.5`, `visual_quality_profile: experimental_numeric27_ade05_v1`, `trajectory_admission_profile: pdms_gt95_v1` in an independent campaign. Existing `python -m driversi.rsi.experiments.reactive_autoloop --config ... --phase production` passes the skill explicitly through selection and seals it. Reuse requires exact source, skill/version, seed, perturbation and native code hashes; all-DriveRL executions cannot be reused as this method.

The existing pipeline performs actual teacher generation -> measured ADE -> same-branch GS/source reconstruction -> numeric admission -> native ChainFlow stage1 export -> new image features -> paired training and full navtest feedback only after every selected source has at least10 unique admitted dataset packages. Both orchestration and the direct ChainFlow family training entry verify this condition. Different export paths for the same package cannot fill the quota. Skills record the actual controller identities. No new dashboard implementation is required.

## Experiment

`exp/history-perturbation-tts-v1/` stores configuration, actual trial receipts and failures. The pilot uses two training-source assets to validate the algorithm; asset selection is not evidence of ChainFlow weakness at the new anchors. `cohort-config.json` expands the verified27dB source to10 samples using64 explicit references and a four-hour production budget, revalidating the two completed final-method branches before reuse. Its first24 references retain seed/control identity; appended references add longitudinal and smaller lateral offsets. Failed and near-duplicate actual trajectories remain excluded; shortages prevent training. The previous24dB production campaign remains independent. No new training result is claimed without a native receipt.

Reference: https://arxiv.org/html/2511.23369v3

## Bounded physical prefetch

`produce(..., prefetch_workers=4)` optionally precomputes at most four physical jobs in isolated spawned processes. The caller requires a `__main__` guard. The same sealed assignments are consumed in original order; actual ADE, GS, export and generated-index writes remain in the coordinator. Default1 preserves sequential behavior. Already admitted samples and immutable outcomes are replay-accounted. Unconsumed lookahead (up to4 when the current row is reused) is settled and reported before training; it is not certified sample yield. Per-candidate locks protect resumed prefetch ownership. Interrupted scheduler attempts must be attributed separately from teacher quality failures.

The active acceleration resumes the existing cohort from `cohort-prefetch-runner.py`; its launch/migration receipts and measured conditional ETA are in `acceleration/`.


## Resident generated-history feature stage

The optional native feature worker is owned by `src/data/catalog/feature_worker.py`. It is activated per generated database by an adjacent `DATABASE.feature-worker.json` descriptor; absent this file, the original subprocess path remains active. The descriptor names the source catalog, encoder profile, worker evidence directory and GPU. Production activation is recorded under `exp/history-perturbation-tts-v1/feature-worker-acceleration/`.

The worker performs full checkpoint/profile validation once, then verifies the exact input file set and filesystem identity/size/mtime/ctime before reuse. It retains the unchanged native encoder. Pixels, source membership, native sample admission and canonical indexing are still checked per job. This metadata reuse assumes a cooperating operator and stable initial weight files, not an adversarial filesystem. No camera, pooling, dtype, model, loss or score change is made.

Requests use a private local file queue. The existing native stage owns job/launch/exit/result receipts. Completed resident RPCs cannot be replayed; resume uses the native stage receipt. A started job without a result remains unknown and cannot be retried automatically. Never-started queued requests may continue after normal worker retirement. An independent watchdog terminates the isolated worker group at job/lifetime deadlines or proxy cancellation, allowing a two-second TERM-to-KILL grace. Idle workers release their GPU after the configured idle timeout.

The initial measured full feature/index stage took109.06s cold and56.35s warm on two existing packages; packed vectors and trainable/searchable status matched exactly. This is a component test, not whole-pipeline throughput, and creates no new production samples. GS/model-export residency and asset-wide reference render reuse remain separate work.

Official ChainFlow split parsing additionally reuses up to two exact byte versions. Each call still reads the complete official YAML; a content change forces a new parse regardless of file timestamps. Member sets are returned independently. Combined real-stage verification measured103.696s cold and48.575s warm; this does not include physics, rendering or training.

### Resident membership cache and bounded top-up

The feature worker retains a fully verified training membership map on a read-only SQLite connection. Database identity, `PRAGMA data_version`, and freshly hashed official split files control reuse. A change rebuilds full coverage; a failed initial read or scan closes the retained connection. Two existing sample jobs measured97.031s cold and12.650s warm in `feature-worker-acceleration/real-validation-v4/report.json`, with byte-identical vectors and zero resume jobs. This is component timing, not total synthesis throughput.

A separately frozen top-up may provide `trajectory_family.proposal_seeds`, one unique integer in `[0,2**31)` per reference. The default remains `seed_start+i`. Exact old successful controls and seeds may be carried first using authenticated generation receipts; their canonical package IDs remain unchanged. All new references are compared with retained successes by the same full-window and future ADE gates. The original wave and its failures remain immutable; reused samples are not reported as newly generated.

## Prepared parallel GS/export batches

`python -m driversi.rsi.experiments.postprocess_pool --manifest /absolute/path/batch.json`
processes an explicitly bounded batch of already executed physical branches. The version1 manifest supplies `output`, `gpus` (distinct1..5, at most4 lanes) and up to64 `jobs`. Each job contains frozen `config/config_sha256`, `plan_sha256`, `generation` arguments, and separate sibling `directory` (job receipts) / `admission` / `generation.output` directories under `exp`. `generation.render` must be true. The caller supplies the existing `diversity.json`; the entry re-derives actual trajectories, verifies retained evidence and repeats within-batch ADE deduplication before dispatch. See the real two-job example in `exp/parallel-synthesis-v1/benchmark/prepared-batch.json`.

GPU slots are held until ordered result consumption. Rendering and source comparison use the same operational GPU, separate from the original sealed physics plan. Native export shares that lane; fresh features use the matching serialized resident service (GPU0 for production). Production training GPUs6,7 are excluded. The fixed input manifest bounds work; this entry does not yet reserve automatic campaign targets/render quotas or replace the running cohort scheduler. All submitted tasks drain before return. Completed jobs validate input, evidence and native sample seals; an unsettled launch cannot be retried implicitly. The worker terminates after owned-child cleanup if its supervisor dies.

Benchmark jobs use `quarantine: true`: config.output owns a dedicated exp subtree, generated_catalog must be that subtree's `features.sqlite`, and `quarantine.json` explicitly declares isolation. Native exports are noncanonical, outcomes are `benchmark_sample_verified`, and no benchmark package enters the production admitted index. Those searchable vectors exist only in the isolated benchmark database. Production package IDs and feature IDs may differ across paths; content parity, not ID equality, verifies equivalent computation.

The two-lane benchmark runs GS, source comparison, native export/admission and fresh-feature ingestion on identical physical controls. It excludes physical preparation, reconstruction, failed attempts, training and evaluation. Initial two-job parallel218.699s versus sequential248.520s had different service startup states; it is not a balanced speedup claim. Exact120 new-view images, all71 actual states/70 actions per branch, native geometric labels and feature bytes were compared. Repetition and final timing receipts live in `exp/parallel-synthesis-v1/benchmark/`; no eight-GPU or full-dataset ETA is implied.

The production top-up independently reached10 unique packages (7 authenticated reused +3 new). Its old/new assignment ID collision was fixed by validating a unique sealed old-plan bridge, comparing all controls/context except routing IDs and preserving the original scene identity. Existing paired training resumed via `cohort-topup/training-resume-launch.json`; it did not migrate into this new batch executor.

Verified warm repetition: two completed jobs in127.924s versus serial248.520s (1.943x observed throughput, 48.5% less wall time). Both worker receipts report model reuse; physical controls,120 pixel files, native labels and vectors match. A middle repetition naturally restarted an idle-retired service and took222.552s; it is preserved as cold, not hidden or presented as warm. This is a two-control component test, not a statistical scaling result. Final evidence: `exp/parallel-synthesis-v1/benchmark/summary.json`.

## Maintained command entry

See [source ownership and entry points](synthesis-source-entrypoints.md). `scripts/rsi/run_synthesis_campaign.py` invokes the existing campaign implementation; `physical_prefetch_workers` explicitly enables1..4 physics workers (default1). The resident prepared-batch supervisor is now source-backed and remains an isolated benchmark, not automatic production postprocessing. Historical launchers are evidence only.
