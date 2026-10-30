# Per-source trajectory expansion

The expansion unit is a **qualified synthetic sample**, not a search attempt.
The initial target is 20 samples per selected training source. Sources are
retrieved from the training catalog; this entry never runs a navtrain student
score and never reads a test scene.

## Executable pilot

`scripts/rsi/run_branch_expansion.py prepare` accepts an existing native DriveRL
teacher task and its matching GS render/source-comparison task. It freezes the
source, assets, native checkpoint, implementation hashes, budget and candidate
grid in `exp/<run>/plan.json`. The initial grid contains one native closed-loop
arrival and 25 bounded jerk/steering-rate arrival controls. All candidates then
use the same frozen DriveRL future policy. This is bounded augmentation, not
learned multi-agent self-play.

`run --output exp/<run>` dispatches up to two independent Harness supervisors.
The coordinator owns their lifetime. It verifies all prior completed artifacts
and rejects unresolved running tasks before dispatching new work. A repeated
run reuses completed tasks without silently relaunching failures. No weights,
losses, optimizer, native scorer or admission threshold are changed.

The generation report separates:

1. attempted physical branches;
2. teacher-qualified branches, including original PDMS >= 95 and physical checks;
3. distinct qualified histories and futures;
4. completed GS render/source-comparison tasks;
5. independently visually admitted samples;
6. sealed native AUGUstus joint-supervision caches.

Only the last count satisfies the sample quota. Insufficient accepted samples
produce `completed_with_quota_shortfall`. A render quota is an explicit pilot
cost limit, not permission to count unfinished samples.

## Diversity and lineage

Each branch executes two seconds to reach a new anchor and five seconds of
future physics. Four history states and the model's selected eight (or explicitly
configured ten) future states are compared in the same source coordinates.
The initial engineering filter requires pairwise history position RMS >= 0.15 m
**and** future position RMS >= 0.5 m. The thresholds are explicit plan fields.
They are not proof of independent situations or different failure causes.

`--existing-rollout` excludes branches already published in earlier batches of
the same source window. Changing the last unused second alone cannot make an
eight-point AUGUstus sample distinct. Candidate order is fixed before execution.
No low-quality branch is accepted to fill a quota.

GS and native label admission use the existing independent procedures. After
pixel review, same-branch prediction-language review and native cache publication,
`admit --sample-report <report>` verifies the sample against this exact batch's
selected branch, rendered images and source comparison. External rerenders and
duplicate sample IDs cannot inflate the count. Repeated admission and resumption
preserve and revalidate the accepted sample list.

## Scaling and benefit claims

3,000 sources times 20 admitted variants is 60,000 dataset rows, not 60,000
independent source situations. Budget with measured yield:

`attempts ~= target_samples / (physical_acceptance * diversity_retention * visual_and_label_acceptance)`.

Render only after physical and diversity filtering. Preserve source groups for
sampling, uncertainty estimates and training/evaluation separation. GS assets
should remain resident in later throughput work; this initial implementation
deliberately reuses the verified per-task renderer.

Increasing dataset size alone does not establish why an earlier pilot had no
benefit. A training comparison must include enough positive-learning-rate updates
and synthetic exposure beyond warmup, the same parent/optimizer/update budget,
and a fixed official evaluation protocol. Report benefit against ordinary
continuation as well as against the parent. Do not credit skill memory with
benefit unless the executed skill plan and branch lineage are bound to the
training receipt.

The initial pilot uses a previously verified asset-covered training source to
test the production mechanism. It does not establish that this source is an
AUGUstus weakness or that 20 nearby variants will improve driving.
