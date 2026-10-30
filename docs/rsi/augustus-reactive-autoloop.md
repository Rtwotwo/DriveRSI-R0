# AUGUstus: actual-history automatic pilot

This finite experiment uses the existing Harness and native model service. It does not create a recurring chat monitor. Production runs on 32341; the full parent and candidate benchmark runs on 32000. All durable outputs are under `exp/augustus-reactive-auto-v1/`.

## Executed boundary

1. Require the completed, unique, exact 12,146-token native navtest parent report. Partial means are not accepted.
2. Prioritize zero and then low PDMS queries in the private test vector catalog. Retrieve only official navval members of the training catalog. Visual similarity is a candidate hypothesis, not proof of student failure.
3. Intersect training candidates with actual reconstruction time support. Require three real source-history frames before the two-second arrival, fifteen source poses from 0 through 7 seconds, and four cameras. Prefer the already rebuilt Singapore asset when the same token has multiple assets; this never grants admission.
4. Execute DriveRL new history and teacher future through typed Harness tasks. Native PDM scores the four-second future; all seven seconds and the additional fifth future second retain physical execution evidence. Rejected/unknown executions remain recorded.
5. Render the exact sealed physical branch, plus source-pose comparisons. Cheap terrain, registration, and existing 25 dB experimental checks run before any paid review.
6. A bounded independent pixel critic examines target/source/box overlays and temporal/multiview context. Both material generation and final admission cover portable moving actors. Unknown is not pass. Credential values stay in the private API directory; request caps, rate limits, response limits and exclusive claims prevent silent repeat charges.
7. Produce native AUGUstus supervision: four history frames, eight future targets and fixed same-branch predictive language. Archive the ten actual future points; do not change the native model or loss. A separate pixel-language review may reject fixed labels but cannot rewrite physical trajectories.
8. Encode only the actual new history pixels with the native visual encoder. Store generated records separately. Unadmitted records are quarantined and excluded from generated search. Cache reuse requires identical export/pixels/profile and current admission. An audited SQL-only reader change retains a distinct producer profile, with an explicit compatibility bridge; model/weights/preprocessing/runtime changes are rejected.
9. Compile same-parent, same-budget original-data and original-plus-generated native SFT arms. Each arm executes four updates, four resumed updates, and an independent replay of the resumed segment. One-source pilots use batch size one in both arms; no fabricated duplicate records.
10. After both arms finish, the 32000 worker runs all 12,146 parent-protocol scenarios for each candidate. Parallel chunk workers change scheduling only, retaining seeds, batches, input caches, native inference and scoring. Parent predictions are reused only from the same completed 32000 baseline.
11. Record paired outcomes, per-sample safety regressions, actual generated-sample exposure, cross-process recovery and service memory. A single adaptive trial cannot automatically establish stable improvement or promote a model.

## Entry points

Use the frozen experiment configuration, not private service configuration copied into the repository:

```sh
.venv/bin/python -m driversi.rsi.experiments.reactive_autoloop --config exp/augustus-reactive-auto-v1/campaign-config.json --phase production
.venv/bin/python -m driversi.rsi.experiments.reactive_autoloop --config exp/augustus-reactive-auto-v1/campaign-config.json --phase evaluation
```

Each role enforces its declared hostname and an exclusive process lock. The evaluation role waits for real native training receipts. The production role is bounded to 24 private queries, at most 16 training candidates, four rendered branches and two admitted samples for this pilot. Private VLM configuration applies the shared request/output/rate budget. Current schema permits at most four admitted samples, not an unbounded dataset campaign.

## Failure and recovery

Completed typed stages and feature payloads are verified and reused. Rejected outcomes are immutable. An unknown subprocess/request outcome is not retried as a new charged or mutating operation. It requires reconciliation or a new explicit attempt directory. A lack of admitted generated samples ends this experiment with `no_qualified_generated_samples`; it cannot fall back to original-image relabelling. Failure cases remain inspectable.

The generated-vector reader verifies both payload checksums and the current native sample admission. Quarantined pixels cannot become searchable because their teacher scored highly. Training, candidate evaluation and gain review require actual outputs rather than launch success.

## Current evidence and limits

The 32000 SFT parent completed all 12,146 samples: PDMS 90.4706129/100, 381 zeros, 6,359 below 95. This differs from the earlier 32-sample smoke result; it is not a comparison to ChainFlow's 94.05 checkpoint.

An independent navval infrastructure pilot executed and restored a real 7-second DriveRL branch, rendered 60 novel-view and 60 source-pose images, prepared native supervision, and encoded 12 fresh history images (2048 dimensions per view). Its 26.9605 dB source reconstruction still failed terrain/object admission; its feature entry remains quarantined. This proves those executable stages, not successful synthesis admission or learning gain.

The current automatic campaign is consuming the completed full baseline. Its live outcome and eventual learning receipts, rather than this document, determine completion. No successful paired AUGUstus training or new-model improvement is asserted before those receipts exist. Test-feedback-directed selection is adaptive evaluation; test images, labels and tokens never enter a training sample or a public gallery.

## Code ownership

- `src/rsi/experiments/reactive_*`: finite orchestration, candidate choice, typed generation, native pair and recovery.
- `src/rsi/experiments/navval_batch_retrieval.py`: shared centroid scan with the same aligned-view reranking rule.
- `src/rendering/diagnostics/gs_sequence_critic.py`: bounded independent evidence producer; existing visual admission remains authoritative.
- `src/data/catalog/derived_features.py`: new-pixel encoding, quarantine and admitted generated search.
- `agents/augustus/candidate_dispatch.py`: same-host parallel native candidate evaluation.
- `src/config/visualization/platform.json`: registration of real receipts in the existing workbench. No copied experiment HTML.

## Historical full-training revision (v3/v4; superseded below)

The latest production configuration is `exp/augustus-reactive-auto-v1/full-training-campaign-v4-config.json`. It supersedes the navval-only pilot without changing its historical evidence. The same `reactive_autoloop` production/evaluation entry points consume this configuration on32341/32000.

- Membership: official navtrain103,288, containing the navval subset. Current ready-vector count103,044;244 are not ready. Do not call this100%vectorcoverage.
- Full baseline:12146 verified rows;381zero,310nonzero<0.6. First wave interleaves64zero/low queries, preserving the691-query queue. This is not completion of all query mappings.
- Vector retrieval traverses available vectors across complete official training membership. Historical structured text/BM25 is optional supplementary evidence. No reviewed VLM caption corpus is assumed and no unsupported Recall@K is reported.
- Full-corpus mapping is separated from asset readiness. Full source-coverage inventory currently202unique anchors from19logs. Missing-asset matches remain in `selection/pending-assets.json`; unattended reconstruction is not implemented.
- `driverl_full_window` uses native DriveRL for all70executed actions from0to7s. An explicitly declared causal resampling of observed2Hz past initializes the10Hzhistory; those interpolated states are not additional sensor measurements. No external warmup or PDMfuture substitution occurs. The native TTS candidate search remains part of DriveRL. If a separate TTOcontinuation is introduced later, its ownership and seam must be recorded.
- Native PDM still scores4s after the2sarrival;5sactual future and the complete7sphysical branch are checked separately. It is not a7sPDMbenchmark.
- Source reconstruction must exceed25dB, plus the existing terrain, contact, camera, object, temporal and visual checks. SourcePSNR is not novel-view ground truth.
- Rendering attempts reserve budget before execution; later source-comparison failures do not refund it. Actual admitted pixels produce fresh features and historical structured text in the generated partition. `derived_features.search` and `search_text` exclude quarantined entries and revalidate native admission.
- This finite wave allows16physical candidates,4render attempts and2admitted samples. Only admitted samples start same-parent, same-budget native SFT followed by full12146evaluation for each arm. Executable skills and outcome/cost memory record attempted actions; no claim that a prompt creates a learned driving skill.

The completed full-native infrastructure pilot has score58.3333/100 (TTCfailed), no contact records and exact restoration. It was rejected before training. A successful handoff or search API is not evidence of learning benefit.

The v3production worker consumes `vector-production-mapping-v2`, a frozen projection of the completed64query vector results. The user explicitly allowed omitting text retrieval while reviewed captions are absent. Historical-text fusion may finish independently; it is not required for this wave. The vector queue preserves902unique training candidates; asset readiness is checked afterward.

After v3completed with two rejected candidates, v4automatically maps all691priorityqueries (reusing64cachedqueries) and excludes the two already attempted training anchors. Its remainingbudget is14physical candidates,3render attempts,2admittedsamples. V3results:96.2409PDM/sourcePSNR24.1214 with unresolvedbinding;100PDM with21arrival/50future contactrecords. Neither startedtraining. Fullfuture score alone cannot override worldvalidity.

## Current full-training automatic loop

The active bounded experiment is `exp/navtrainval-full-loop-v1/campaign-config.json`. Use the same production/evaluation module entry points above with this path, on 32341 and 32000 respectively. It consumes the existing completed official SFT parent and its full 12,146-scene baseline rather than repeating identical training or scoring. Experiment state, failures and new checkpoints stay beneath this run's `campaign/` directory. This is a full-population retrieval with bounded production/training, not full-103k SFT.

- The native training catalog has all **103,288** official members and vectors, including audited supplements. The user calls this navtrainval; membership is defined by the official manifest, not a newly guessed union. Existing irregular-timestamp flags remain and retrieval eligibility never grants temporal training admission.
- `full_training_mining` reuses all **691** matching frozen vector queries, constructs an FTS index over training-history facts, and preserves visual top-8 plus text top-8 through reciprocal-rank fusion. Current completed mapping has 6,265 training candidates and 18 with full-window assets. Reviewed VLM captions are absent, so the text method is explicitly structured history text. Recall requires separately annotated relevance; it is not inferred from coverage or similarity.
- Production can explore up to 16 anchors within 24 rendered branches and 20 admitted samples. Each anchor targets 10 distinct samples from at most 24 native DriveRL seed attempts. The frozen 0.5 m full-window/future ADE rule remains an experimental deduplication criterion, not a SimScale claim. Reaction/Recovery coverage is a motion-based hypothesis, not a proven failure cause.
- The optional `explore_until_global_limit` budget mode permits more anchors than all targets can simultaneously fill. Shortage remains explicit. Optional consecutive-duplicate, visual-failure and execution-failure budgets transfer work to other anchors; they do not establish that a scenario is impossible. Completed receipts remain immutable and owned pending work settles before a family is skipped. A smaller admitted cohort can enter a bounded training pilot without claiming the ten-per-anchor objective was achieved.
- `production_gpu` routes rendering, native postprocessing and paired training to the same declared device (default 6 for historical callers; this run uses idle GPU7). DriveRL presently executes on CPU. This is a single-GPU native training pilot, not a claimed dual-GPU training run. Full candidate evaluation remains on 32000 after the two arms and their recovery/exposure checks finish.
- Near-plane physical boxes are clipped before distorted-camera projection. Seventeen edge samples approximate the overlay; the recorded failing calibration is checked against dense sampling. Clipping does not certify appearance, remove actors from audit or change physical trajectories. All original visual and PSNR>25 dB requirements still apply.
- New admitted pixels must be encoded again. The audited `vectors.py` provenance-only migration has an exact-SHA compatibility bridge, while weights, runtime, cameras and preprocessing remain exact matches. The actual producer profile is retained separately; unknown implementation changes fail closed.
- Generation outcomes enter Harness execution memory without learning credit. New training samples use native AUGUstus supervision/loss, matched control/treatment budgets, cross-process replay and positive-learning-rate exposure checks before full navtest reevaluation. Gain, regressions and promotion remain empirical outcomes, never guaranteed by successful dispatch.

The unified workbench registers mapping, production, cohort and evaluation receipts under `navtrainval-full-loop-*`. Only allowlisted aggregate values are projected. No new dashboard template, public test identifiers or credentials are exported.

## Synthetic sample storage

New exports use `logs_sensors_history_future_v2`. One branch remains one native chronological log and an independently sealed sample; shards can be indexed through the existing generated-vector/catalog receipts without duplicating image payloads.

```text
native/
  logs/<branch>.pkl
  sensors/history/<branch>/<camera>/<frame>.png
  sensors/future/<branch>/<camera>/<frame>.png
  manifests/<branch>.window.json
  export.json
```

`logs/` contains only native pickles: the official unfiltered loader attempts to unpickle every file there. The manifest records four historical frames, ten archived future points, timestamps, physical ticks, four-camera image paths and checksums. Per-frame calibration, object annotations and ego state stay in the native log. `export.json` binds source membership, physical branch, rendering and all copied pixel hashes. The generated feature database references this same admitted sample; unadmitted diagnostic exports remain quarantined.

This preserves a single sensor root for native consumers. Only the first four history frames enter current policy observations and history image features. The unchanged AUGUstus adapter selects its native eight future targets; all ten executed future points and forty future images remain archived for other consumers. No future frame becomes a policy input because it shares the native chronological log. The original 0-second diagnostic capture and dense 0.1-second physical rollout remain in their sealed simulation/render artifacts; this training export is the 2 Hz four-history/ten-future window.

The first diagnostic layout put JSON in logs/ and failed native unfiltered discovery. It is retained under the experiment with an explicit rejection; version2 moves JSON to manifests/ and has been tested with the real AUGUstus SceneLoader. Layout verification is not visual or training admission.

## Canonical dataset root

New data are published to `dataset/synthetic_navsim/samples/<content-id>/` under DriveRSI. Each sample contains the native `logs/`, `sensors/history/`, `sensors/future/`, `manifests/` and `export.json` structure above. The original experimental `native/` location is a stable symlink to the canonical package. Experiment reports, HF training caches, checkpoints, diagnostics and GPU logs remain in `exp/`; earlier sealed data are not relocated.

Copying uses a per-identity lock and a staging directory on the same dataset filesystem. The complete native log, pixel files, source references and hashes are validated before a directory rename exposes the package. Interrupted staging is retained; a retry creates another staging directory rather than exposing partial files or modifying a completed sample. Repeated publication of the same identity revalidates and reuses the existing package.

Presence under `samples/` does not imply training eligibility. `admitted/<package-id>/<receipt-id>.json` is created only by the feature worker after native trajectory/visual/language admission and newly encoded history features match the same sealed sample. The original generated-vector partition retains its own admission checks. Training consumes explicit admitted native reports, not an indiscriminate directory glob. `dataset/` is excluded from source Git tracking.

Multiple experiments can reuse the same immutable pixel package while appending separate admission references. Repeating the same sample/feature evidence returns the same reference; a new experiment never overwrites an earlier admission record.
