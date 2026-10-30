# Frozen-policy skill evolution in DriveHarness

This pilot evolves executable prompt skills while AUGUstus remains frozen. The private Doctor model identifier and request configuration stay fixed, and no Doctor training is performed; provider-side weight immutability is not witnessed. It runs on real logged observations and native planning scores. It neither trains weights nor synthesizes new camera observations.

## Executed method

1. Freeze the parent bundle, native action path, observation cache, scorer, paired seeds and validation membership in a declaration.
2. Route each observation using current speed and the navigation command. The fixed router chooses a conditional skill node; it does not diagnose the failure's cause.
3. Select the node with the lowest development mean PDMS. Send its current text, observed state, metric feedback and persisted prior attempts to the Doctor. No validation examples, scene tokens, future trajectories or target coordinates enter that request.
4. Accept only a bounded JSON edit of that node's English text. Numeric target actions, scene identifiers and control markup are rejected. Responses are data, never executable code. Cache request intent before transport and the response after transport; never automatically resend an unknown outcome.
5. Seal the candidate. Run native, incumbent and candidate prompts with the same frozen model, image/history inputs, action decoder and paired seeds. Save every output before calling the unchanged native scorer. There is no oracle selection of the best trajectory.
6. Evaluate on the development cases and a new validation cohort drawn from other original acquisition drives. Promotion requires development improvement, positive validation transfer, no mean validation decline versus native or incumbent, and no new zeros or safety/comfort regressions on either cohort. Failure is a completed rollback decision, not an execution error.
7. Commit the decision and active skill in one SQLite transaction. Rejected edits remain negative experience. The next Harness episode reads this state, selects a node and asks for another bounded revision automatically.

The first pilot has two episodes, two Doctor calls maximum, 32 development scenes, eight new acquisition drives per validation episode, two seeds and three arms: 480 predictions. Development comes from a deliberately difficult prior pilot. Validation membership is chosen by a stable hash without reading score values. Different segments of one original acquisition are treated as the same group. All these navtest cases are now adaptation/validation evidence, not a pristine final benchmark.

## Implementation

- `src/rsi/decisions/skill_evolution.py`: proposal contract, paired admission and transactional memory.
- `src/rsi/decisions/skill_doctor.py`: isolated development feedback and cached private API transport.
- `src/rsi/experiments/skill_evolution.py`: membership, episode preparation and durable decisions.
- `src/rsi/plugins/skill_evolution.py`: typed services registered in the existing Harness DAG.
- `agents/augustus/skill_inference.py`: native frozen inference with explicit pilot/evolution comparison modes.
- `src/config/rsi/policies/augustus-skill-evolution.json`: bounded experiment policy.

Run artifacts and memory live under `exp/augustus-skill-evolution-v1/`. A fresh campaign has a fresh registry/output namespace; it does not overwrite previous pilot evidence. The registered DAG is a finite automated experiment, not a background monitor. Completed-task replay is rejected by the existing engine. Unknown API or scoring outcomes retain their receipts and are not silently retried. Transactional memory replay cannot create a second promotion.

The earlier fixed-skill pilot is pinned to Git `b30e71eacd12318594e291997f2bcfed429d4202`. Its declaration correctly rejects a newer checkout. Reproduction requires that recorded source revision, not rewriting the old plan's hashes. The first evolution run's implementation archive and declaration similarly preserve the executed revision; post-run resilience fixes must be identified separately.

## Evidence and limits

Model parameter versions, gradients, checkpoint hash, exact same-seed replay and action equality with/without post-hoc explanation are checked by the native inference witness. Current/history observations are an explicit model-input allowlist; cached future targets are excluded. No optimizer exists in the inference path.

The mechanism is demonstrated only if the actual Harness records show proposal, sealed inference, independent scoring, decision, and subsequent experience consumption. A rollback is a valid outcome; a passing synthetic unit test is not driving evidence. A local experimental skill promotion never changes production policy.

This does not prove that memory improves future performance over a Doctor with no memory. That requires equal-budget controls, more tasks and a frozen held-out evaluation. The new candidate/ incumbent comparison does not include a matched-length neutral arm; the earlier fixed-skill pilot did. Small score changes can cross discrete metric boundaries. Two seeds and eight validation drives per episode provide a bounded smoke experiment, not a statistical guarantee against regression.

The Doctor currently sees metric aggregates and up to six difficult token-seed observations containing speed/navigation; it does not see images or actor geometry. Its failure explanation is a hypothesis. Node selection prioritizes lowest absolute incumbent PDMS, not largest regression. Previous validation veto/pass reasons enter experience, so each subsequent validation cohort must be fresh. Validation examples and per-case validation scores never enter the Doctor request.

## Caption-assisted continuation

The optional `caption_library` input adds current-anchor attribute evidence through `src/data/catalog/skill_context.py`. Only development rows are read. Token/log/timestamp provenance is checked privately; the Doctor sees allowlisted attributes, camera references, uncertainty and a command-conflict flag, without scene identities, coordinates, free caption, retrieval tags, original training answers or caption navigation. Native speed and navigation stay authoritative for this controlled comparison. These are unverified annotation claims, not causal truth or instructions. The fixed speed/navigation skill router is unchanged; semantic routing effectiveness is not claimed.

The Doctor has offline four-camera whole-image context; the student keeps native cropped front/left/right history. This is explicit diagnostic observation augmentation, not an equal-observation caption baseline. Only a generic conditional skill edit reaches the student. Identical seed pairs are grouped into one scene example before selecting six difficult cases; rejected candidate feedback includes paired incumbent deltas.

`initial_memory` replays immutable prior events into a new campaign namespace. This continuation supports the current rejection-only history with the original preset still active. Cross-campaign inheritance after a promotion requires additional baseline/initial-preset handling and is not supported here; promotion and subsequent use within a campaign are supported. Prior aggregate validation veto reasons may enter experience, but validation examples/captions/per-case scores do not.

The caption continuation discovered only six unused acquisition drives among the 44 represented by the historical full evaluation. It therefore predeclares two rounds of 32 development scenes plus three fresh validation drives, two seeds and three arms: 420 predictions. The small validation cohorts are exploratory. Subsequent campaigns cannot silently recycle them as fresh validation.

Use `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting the existing AUGUstus Python environment, as required by deterministic CUDA execution. The first caption-assisted attempt omitted this setting and failed before any saved prediction. Its completed Doctor response and failed task remain intact. A fresh attempt uses `cached_doctor_replies` to adopt an exactly matching completed request/response without issuing another paid call. An existing unknown intent is never replaced or resent; an unmatched cached job is rejected.

The existing private platform accepts `current_skill_campaign` in current-only mode. It displays aggregate context coverage, command conflicts, stage receipts, skill text, paired means, safety vetoes and experience counts. It exports no test identities/captions and ignores stale synthesis live snapshots while showing this run. The public site remains closed.

If this mechanism yields reproducible transfer, the next method study should compare native, fixed skills, Doctor with reset memory, and Doctor with persistent memory under the same request and inference budgets. Freeze the winning system before final evaluation. Continuous visual execution after leaving the logged trajectory still needs new sensor observations or a simulator.

## First real execution

The registered two-episode task completed in 586.07 seconds, scoring all 480 predictions and making two private Doctor calls (4,223 input and 507 output tokens). Both responses passed the proposal contract. Both edited the following/speed node automatically. Episode two read the first rejected edit and its development feedback from durable memory.

Neither candidate passed admission. Development PDMS changed from the fixed incumbent's 49.4791 to 47.6693 and 47.6575 points. Both introduced the same collision-related zero in one scene under both seeds. The disjoint validation cohorts showed only small positive mean changes (0.0398 and 0.0034 points versus incumbent), which did not override the development safety veto. The active preset remains `observed-prefix-v1`; two rejected experiences are retained.

This is evidence of autonomous feedback processing and safe rollback, not a successful learned improvement. The Doctor's current metric/state-only context is a major method limitation. Subsequent research should ground failure diagnosis in observed history and relevant event evidence before claiming an ability to develop reliable new skills.
