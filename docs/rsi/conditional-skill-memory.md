# Conditional skills and controlled external memory

This extends the existing frozen AUGUstus Harness, not the native model. `grounded_conditional_v1` proposes one skill's text and current-state eligibility. No SFT, optimizer, action-coordinate editing or scorer changes are introduced.

## Evidence and executable conditions

Doctor feedback contains anonymous development observations, measured native/incumbent/candidate metrics, native current state and unreviewed current-anchor caption attributes. Safety regressions are prioritized and up to two higher-scoring examples provide contrast. Every diagnosis names supporting evidence IDs and records a hypothesis, counter-evidence and uncertainty. The validator checks actual references and at least one measured fact; it does not certify a causal explanation.

Eligibility is limited to four native commands, a current-speed interval and optional STOP exclusion. The numbers select applicability, not target actions. The existing node router remains unchanged. When a gated node is disabled, its prompt is exactly native and inference verifies both raw and rounded trajectories equal the paired native output. Initial and untouched nodes preserve the old unconditional behavior. No future trajectory, score or caption controls online eligibility.

## Matched retained/reset conditions

Both conditions start from the same experimental skill preset without earlier event history. Each receives two proposal opportunities on the same 32 development anchors, eight validation drives per round, paired seeds 42/43 and native/incumbent/candidate arms: 480 plans per condition. The first Doctor response is reused exactly, and comparison checks first-round prompt and raw-action equality. There are three distinct API requests at most, not four; adopted-response tokens are reported separately.

Retain reads its preceding event. Reset hides event history and removes obsolete candidate/incumbent comparisons before constructing the next request. Both retain the active promoted skill and current incumbent development evidence. This tests the incremental role of explicit attempt history, not absence of all learned external state. A rejected edit still completes its Harness stage and permits the next episode.

All original acquisition drives have appeared in prior trials. New validation anchors exclude prior selected anchors and all development drives. Round validation drives are disjoint, but reused from prior validation; this is a bounded navtest adaptation comparison, not pristine held-out generalization. Two rounds cannot statistically establish persistent-memory benefit.

## Execution and recovery

Use the unchanged `.venvs/augustus` environment. Existing environments are warm-reused and every inference process executes frozen/determinism witnesses. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before launch. Current authorized setup shares physical GPU 6, runs conditions sequentially and sets a 0.25 PyTorch allocator fraction. This caps the allocator, not total driver/context memory, and does not reserve compute from other jobs.

`python -m driversi.rsi.experiments.skill_memory_compare launch exp/conditional-skill-rsi-v1` launches existing registered Harness tasks. Campaign configs remain under exp and refer to existing private API configuration without copying secrets. Shared-first-reply adoption is exact and unknown API outcomes are never resent. A failed condition is preserved; the supervisor does not restart it indefinitely. Each Harness task retains its declared finite wall/compute budget.

`python -m driversi.rsi.experiments.skill_memory_compare compare exp/conditional-skill-rsi-v1` audits completed reports, pairing, zero weight updates and experience visibility. Promotion uses the unchanged strict development/validation gain and no-new-zero/safety-regression gates; it only updates the experiment's active skill, never production policy.

The private existing platform displays the two conditions, eligibility, native/candidate scores, decisions, counts and budget scope. No test identities or per-test captions are exported. This campaign does not restore app automation or create an unattended recurring experiment.
