# Native EPDMS and generated-data admission

The reusable evaluator is in `src/evaluation/epdms/`, configuration in
`src/config/evaluation/epdms/`, and the additional data gate in
`src/rsi/correction/epdms_admission.py`. The evaluator was adapted from DriveScorer;
`migration.json` records the original source hashes. Runtime NAVSIM and nuPlan
code comes only from DriveRSI's owned `src/vendor/driving/` tree. The existing
metric cache and submission may be read as data; they are not imported as code.
Use a separate process from the v1 teacher runtime.

## Complete metric

The [official metrics](https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md)
define four multiplicative conditions (NC, DAC, DDC, TLC) and five weighted
components: progress 5, TTC 5, lane keeping 2, history comfort 2, extended comfort 2.
The weighted denominator is 16. Extended comfort (EC) compares independently
produced consecutive plans over their overlapping prediction interval. It is
different from the smoothness of one executed trajectory. The native
[SceneAggregator](https://github.com/autonomousvision/navsim/blob/main/navsim/planning/simulation/planner/pdm_planner/scoring/scene_aggregator.py)
aligns the states by their anchor-time difference and applies the original EC
function. The generated-data contract fixes this difference at 0.5 seconds.

The [official one-stage replay](https://github.com/autonomousvision/navsim/blob/main/navsim/planning/script/run_pdm_score_one_stage.py)
can omit EC and its weight when no adjacent prediction exists. That behavior is
preserved solely for benchmark replay parity. A replay covering every navtest
token therefore need not have EC for every token. The command reports these
coverage counts separately. Such a replay receipt is never a generated-data
admission receipt.

## Generated branches

The user-requested policy requires every generated trajectory's full score to
be strictly greater than 0.95. Missing EC, zero-filled missing components,
dummy DriveRL interface paths, or two slices of one executed path cannot meet
that contract. The gate recomputes the full scalar from every native component,
checks the strict boundary, and binds the receipt to the exact physical branch,
context, captured planning evidence and owned scorer source hashes.

Input consists of two seconds of real generated history, a scored four-second
future, and five seconds of actual world observations to support TTC lookahead.
This is not a claim to score all seven rollout seconds as one official benchmark.
Route polygons, traffic signals, ego and other actors must come from that same
executed branch. Original-log human-penalty filtering is disabled because the
counterfactual branch has no corresponding human trajectory. This distinction
and the reached-anchor reference generator are recorded in the receipt; the
result is a counterfactual admission metric, not an official navtest result.

`planning_pair.py` requires a sealed pair of full 4-second state arrays, sampled
at 0.1 seconds. Both initial states must match the actual generated history at
their own anchors. The current array must match the branch being scored. These
checks detect inconsistent evidence; they do not manufacture independent plans.
The arrays must originate from a trusted native planning producer.

DriveRL's default action mode returns a dummy trajectory carrying one control
action. That dummy remains prohibited. `src/evaluation/search/driverl_planning.py`
now provides a **privileged policy-rollout planning adapter**: from each saved
anchor it recursively runs the unchanged action policy inside a cloned world for
four seconds, materializes a state array before the main branch executes, and
restores the world, planner/controller memory, causal history, diagnostics and
random states. Two forecasts add 80 world steps to the execution budget.

This is not a learned DriveRL trajectory head or an official navtest model score.
It uses the same deterministic world as execution, including that world's actor
and traffic-signal schedules. In the first witness both overlap arrays are
exactly equal, so native EC is 1. This measures consistency in the specified
world, not robustness to unknown traffic or prediction error. The complete scalar
is a synthetic admission metric with this declared scope. It must not be reported
as evidence that DriveRL learned independent long-horizon planning. A changing
future traffic-light schedule can be privileged information in this adapter;
the witness's future signal lists are empty, so that effect was not exercised.

The first real witness is recorded under
`exp/continual-scene-curriculum-v2/planning-capture-probe/`: continuous new history,
native PDMS 1, complete counterfactual EPDMS 1 including EC, exact world restore.
This receipt alone grants no visual, dataset, or training qualification.

## Public navtest implementations checked

- [WA-JEPA](https://github.com/AFARI-Research/WA-JEPA/blob/main/scripts/evaluation/run_navsim_epdms.sh)
  invokes native NAVSIM v2 one-stage scoring with `navtest` and non-reactive
  traffic. This is distinct from a navhard two-stage score.
- [SUV](https://github.com/ASH-2046/SUV/blob/main/experiments/navsimv2/run_pdm_score_multigpu.py)
  shards single-scene scoring across GPUs, then concatenates raw outputs before
  building adjacent-scene aggregators and computing final scores. EC pairs must
  not be silently lost at shard boundaries.
- [ViRA](https://github.com/OpenDriveLab/ViRA/blob/main/navsim/planning/script/run_pdm_score_one_stage_gpu_diffusiondrive.py)
  batches predictions, retains simulated ego states, joins neighboring anchors
  from the same log, and calls `SceneAggregator`. Its final aggregation explicitly
  removes the EC weight where native EC is missing, like the official one-stage
  protocol. That fallback remains forbidden for our complete-metric admission.

All three evaluate native predicted trajectories; none establishes that an
action-only teacher's dummy trajectory is an acceptable EC input. Our privileged
adapter and its deterministic-world limitation therefore need their own label.

## Commands and validation

```bash
.venvs/augustus/bin/python -m driversi.evaluation.epdms.evaluate \
  --benchmark navtest --check
.venvs/augustus/bin/python -m driversi.evaluation.epdms.evaluate \
  --benchmark navtest --submission /path/to/submission.pkl \
  --output-dir exp/epdms-replay --workers 1 --smoke --save-raw
.venvs/augustus/bin/python -m pytest -q tests/integration/test_epdms_planning_pair.py
```

The 16-scene stored-prediction replay in
`exp/continual-scene-curriculum-v2/epdms-migration-smoke/` completed without model
inference. Its eight single-frame components match the original full replay
exactly; one EC pair is available in that subset and matches. Subset aggregate
scores are not compared against full-benchmark scores because adjacency coverage
differs. Separate native tests exercise comfortable/disagreeing plans and reject
wrong-anchor, missing, modified and dummy planning evidence.

The old full replay audit found 10,040 EC values and 2,106 missing values among
12,146 scenes in each of three inspected runs. These historical results are
preserved, not rewritten or relabelled as complete-metric synthetic admission.

## Immutable task copies
The native render task copies scored scene and rollout artifacts into its own execution directory. Admission verifies distinct file roles and exact SHA-256 content equality, then rehashes both scored originals and render copies. A path relocation alone is permitted; missing originals, altered data, role mismatch, changed scorer sources, and missing EC remain rejected. Changes to this admission implementation require fresh score receipts; historical receipts are never re-signed.
