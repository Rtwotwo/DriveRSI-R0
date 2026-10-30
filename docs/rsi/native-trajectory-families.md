# Native trajectory families in the automatic RSI pilot

The official training manifest has 103288 members. A registered vector collection includes base and recovered-history feature profiles. The complete 12146-case AUGUstus SFT evaluation is reused. Test-side query IDs and images remain private; only mapped training sources enter generation. A collection change requires new mapping output; incompatible external ranking seeds are ignored, not rewritten. Similarity scores are not Recall@K without relevance labels.

`reactive_autoloop` accepts an explicit `trajectory_family` policy. Initial bounded campaign v5 selects two asset-supported training anchors, each with 24 seeded native TTS attempts and a target of ten distinct qualified samples. Native DriveRL controls all 70 actions; observed past is causally resampled as previously declared. This is not learned multi-car self-play. Seeds are native sampling inputs, not new model weights.

Teacher and render phases are separate. Complete actual 71-state trajectories are compared in the same source coordinate/time system. Both full 7-second and future 5-second ADE must reach 0.5 m against retained branches. A different seed or filename is never sufficient. Near-duplicates are retained as evidence but not rendered again. Physical admission continues to require the existing native 4-second PDMS >=95 and complete 5-second future execution, contact checks and exact restoration. There is no new 7-second PDMS definition.

Reaction_candidate requires observed deceleration in proximity to a forward object. Recovery_candidate requires observed return toward the source path after a lateral departure. These are explicit kinematic hypotheses, not causal certificates or learned skills; unknown cases stay unclassified. No taxonomy quota is filled with invented labels.

GS uses the same physical branch. Strict source-pose PSNR >25 dB and existing object/geometry/temporal checks remain mandatory. Source PSNR is not a claim about unseen-view PSNR. Render limits do not forbid further cheaper trajectory attempts. Qualified generated pixels receive fresh features and same-history text through the existing admitted catalog. Failed or duplicate attempts remain in exp.

Per-anchor distinct/qualified/shortage counts and sealed terminal receipts support recovery. Unknown owned tasks are reconciled, never blindly restarted. The generation-memory database records execution/data outcomes without student-gain credit. All artifacts stay under exp; the existing visualization registry exposes only aggregate progress.

Both native SFT arms share parent, batch size and update budget. The batch divides both data pools to avoid dropped tails. Family experiments allocate enough bounded steps for dataset coverage, then check actual positive-learning-rate sample exposure BEFORE full navtest reevaluation. Missing exposure, expired production budget, or failed admission blocks evaluation/training as appropriate.

This remains a finite paired pilot. The current learning reviewer records full-evaluation gains and failures but does not automatically promote a single trial. Stable skill improvement, multi-modal family success, automatic missing-GS reconstruction, and recursive promotion/recollection still require evidence and further implementation. The current changes do not claim those capabilities are complete.

## Reference-based coverage audit (diagnostic v1)

The current 0.5 m ADE threshold is an engineering pilot setting, not a SimScale rule.
[SimScale section 2.3 and Table 5](https://arxiv.org/html/2511.23369v3#S2.SS3) describe sparse perturbation-endpoint coverage with longitudinal 5 m and lateral 0.5 m steps. It produces perturbed states and then expert futures. The spacing is not a minimum 7-second ADE. Its reward filters also use a different EPDMS protocol/stage; they do not replace this campaign's native PDMS>=95 and source-pose PSNR>25 gates.
[MTR batch_nms](https://github.com/sshaoshuai/MTR/blob/master/mtr/utils/motion_utils.py) suppresses motion-forecasting candidates using endpoint distance after score sorting. It is a reference for geometric candidate reduction, not a validated driving-data admission threshold.

`trajectory_coverage.family_coverage` adds diagnostics to each automatic family summary: 2-second arrival-state separation, full/future ADE, endpoint difference, largest one-second local separation, and smoothed speed difference. Occupancy uses 5 m/0.5 m steps in the fixed initial ego frame with two shifted grid origins. This is neither Frenet projection nor a reproduction of SimScale's interleaved sampler. Boundary-sensitive cell counts never certify distinct samples or fill the target-of-ten quota. Existing sealed receipts are preserved; old descriptors without initial yaw report unavailable coverage.

New production selection policies should separately test arrival-state/history coverage, evidence-supported interaction modes, and near-duplicate geometry within a mode. A different random seed, grid cell, or Reaction/Recovery hypothesis is insufficient. Same endpoint with different slowing/yield timing needs temporal evidence; different futures from effectively the same history also need a compatible supervision strategy. Prefer a diverse set of genuinely different reached states over conflicting labels for an indistinguishable input. Maintain matched paired budgets and verify actual positive-learning-rate exposure; exposure is a consumption check, not proof of sufficient training or positive learning gain.

The live v5 admission and ADE-v1 selector are unchanged. The new metrics are logged for calibration of a separately versioned future policy; they cannot promote a model, certify a skill, or bypass visual review.


## Explicit PDMS/ego-camera experiment and navigation repair

The operator-selected `pdms_gt95_v1` trajectory profile requires native counterfactual
PDMS strictly greater than 0.95. Separate background contacts, direction and world
event vetoes are retained as diagnostics. It does not alter the native scorer,
teacher model or loss. Complete 20+50 physical steps, score availability, changed
history, source identity and exact restoration remain transport/execution checks.
No partial rollout, repeated final frame, or extrapolated path can satisfy them.
Both full7s and future5s ADE remain >=0.5m against previously retained branches.
Legacy default `strict_world95_v1` and historical receipts are unchanged.

`driverl_navigation_v2` is an optional generation-plan setting. Native DriveRL
roadblock correction uses the current pose/map/task navigation, not future poses.
It disambiguates the initial lane using corrected navigation, includes real
adjacent geometry when the rear axle is close to a connector exit, and follows
unique locally supported map connections even if distant map reachability is
unknown. Missing/ambiguous local topology remains a support boundary, never a
fabricated destination. Original and corrected navigation, input hashes and
continuation decisions are saved. A current lane incompatible with the next
navigation block still requires legal lane-change-aware replanning; extending
straight beyond the map or joining arbitrary endpoints is not implemented.

The `experimental_egoview25_v1` visual profile removes missing terrain support and
exhaustive per-object registration metadata as independent hard vetoes. Actual
ego-camera road/lane/on-road-vehicle/occlusion/temporal/multiview/large-hole checks
and mean canonical-source PSNR>25dB remain. Unknown height is recorded without an
invented overlay. Reviews are bound to their exact profile and pixels and cannot
be upgraded to strict geometry certification. See rendering/gs-multiview-training.md.

`review_budget_preflight=true` checks the shared finite VLM ledger before a new
candidate. A branch reserves capacity for 60 camera/time reviews plus one native
language review; it does not reserve an unlimited retry allowance. Exhausted
quota is `review_budget_exhausted`, missing/invalid evidence is
`visual_review_incomplete`, and genuine observed rejection is `visual_rejected`.
None grants training admission. Prior charged failures continue to count when
the operator's finite experiment allowance changes. Concurrent ledger writes
are serialized, and technical failures remain visible to Harness.

Real validation artifacts for the three repaired source scenes are under
`exp/egoview-pdms95-loop-v1/`. They executed all70 DriveRL actions with exact
restoration and scored95.1090/100/99.8290. These are counterfactual native4s
scores with5s actual future, not7s benchmark scores, student gains or promotion.
Automatic production/evaluation uses campaign-v3-config.json and the existing
two-host reactive_autoloop. Its generation target is not a completed training claim.
