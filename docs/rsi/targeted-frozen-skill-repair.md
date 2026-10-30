# Targeted frozen skill repair

`driversi.rsi.experiments.skill_repair` uses the existing typed DriveHarness providers, durable task journal and transactional skill memory. No separate conversational dispatcher, optimizer or renderer is involved.

The bounded first experiment addresses `GO STRAIGHT` observations routed to `following_speed_check` whose measured incumbent TTC fails. This is a score-defined failure family, not a causal label. The native observations, checkpoint, action decoder, seeded generation and scorer remain unchanged.

## Automatic execution

1. Fix development members and cross-acquisition-log validation members before any new prediction. Retrospective native TTC selects the validation failure family; broad controls are score-blind. Prior validation drives may be reused and the experiment is explicitly adaptation.
2. Import previous failed modifications and measurable safety regressions. Doctor sees development evidence only: native state and scores plus unreviewed current-anchor caption claims. Caption navigation never overwrites native navigation; whole captions never enter the student.
3. Doctor proposes one bounded text and native-state eligibility modification, with intended case activation changes and TTC target cases. Validate claimed activation changes against both actual gates.
4. Run five paired arms: native, incumbent, text only, eligibility only, combined. Exact native fallback is retained. Test metric/scene identifiers cannot enter the deployed eligibility rule.
5. Require actual changed action and target TTC improvement; check remembered safety failures against the current incumbent. Choose an eligible variant on development results only.
6. Apply the original admission rule unchanged to full cross-log validation. Separately check TTC transfer on the same-family subset and safety/non-decline on broad controls. A failed proposal is rolled back, not retried indefinitely or silently relaxed.
7. Atomically persist full proposed text/gate, each ablation's actual definition and feedback, chosen/diagnostic variant, regression obligations and verdict. Next round reads this state automatically even after rollback.

The first run is two rounds, 32 development anchors (8 target,24 protection),16 validation anchors (8 family,8 controls),two seeds,five arms:960 plans and at most2 new Doctor requests. Historical failed proposals are evidence, not fictitious accepted revisions. Final trajectories are sealed before scoring; validation scores do not select the output trajectory or choose among variants.

## Entrypoints

Declare using the existing AUGUstus environment (requires native HF cache dependencies):

```
.venvs/augustus/bin/python -m driversi.rsi.experiments.skill_repair exp/targeted-skill-repair-v1/config.json exp/targeted-skill-repair-v1/declaration
```

The existing Harness worker executes the complete declared graph:

```
.venvs/augustus/bin/python -m driversi.rsi.experiments.skill_memory_compare worker exp/targeted-skill-repair-v1
```

After terminal completion, audit:

```
.venv/bin/python -m driversi.rsi.experiments.skill_repair_report exp/targeted-skill-repair-v1
```

Environment/GPU limits are inherited from the validated native runtime. The orchestrator must itself use the native AUGUstus interpreter because stages inherit `sys.executable`; the general `.venv` lacks `datasets`. Unknown API outcomes are not resent. The private platform displays aggregate variant effects and regression verdicts; it exports neither test identities nor caption claims. No public site publishing or SFT is part of this task.

Limitations: command/speed eligibility is still coarse; Doctor has no independent raw-image or temporal event diagnosis in this run. There is no new length-matched/mismatched text control. Two automatic rounds establish execution and feedback inheritance, not statistically established long-term RSI benefit. Experiment promotion does not publish a production driving policy.
