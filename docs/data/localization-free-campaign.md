# Localization-free augmentation campaign

The production selector is `src/rsi/experiments/localization_free_retrieval.py`.
Global budgeting is owned by `src/rsi/decisions/augmentation_budget.py`, with
dispatch integrated into `reactive_autoloop.py` and `reactive_selection.py`.

## Selection contract

- Use the frozen 9405 benchmark's 239 zero-score query identities.
- Reuse the original ungated global visual ranking; take Top3 after continuous
  training-history validation and one source group per query.
- Globally deduplicate training tokens. Do not filter or rank by query/candidate
  coordinates, geographic distance, map identity, absolute heading or GT future.
- Production has 562 seeds from 272 source logs. Each query returns three
  candidates, giving 717 pairs before global deduplication.
- Cluster-visual rankings are only a comparison. The complete leaf candidate set
  is checked before projecting rows to token/cosine; all geographical fields and
  the earlier filtered methods are ignored. The comparison cannot change the
  production seed set.
- Test correspondences stay in `private/`. The production queue contains only
  training-source identities and aggregate support counts.

The previous 25 m / 30 degree selection remains historical audit evidence. It is
not the production recipe. Geographic nearest-neighbor evaluation may still be
reported separately, but it cannot certify interaction similarity. Using navtest
failures to guide development remains adaptive evaluation, even without location
GT or test pixels becoming training data.

## Harness budget

The global manifest distinguishes 562 source scenes, a target of 5,620 qualified
new samples, and a maximum of 32 physical attempts per source. A source stops
after three consecutive duplicate trajectories, one visual failure, or two
execution failures. These stops conserve budget; they do not modify admission or
declare a source physically impossible.

Each native wave contains at most two sources, targets ten samples per source,
and reserves at most 64 physical attempts and 40 branch renders. The existing
native >95 PDMS and >25 dB profile, diversity checks, branch identity and training
admission remain unchanged. The single global file lease prevents overlapping
waves from oversubscribing this campaign. Previous unsettled/failed waves block
new dispatch instead of silently reissuing work. Completed source selections are
not automatically retried in another wave.

Asset absence and prior visual failure are production states, not relevance
scores. The current real run identifies 560 sources needing assets and two
needing repair. Neither is silently replaced by an unrelated asset-supported
source. Updated asset evidence requires a newly versioned campaign declaration.

The bounded mode defaults to a separate training dispatch after qualified data
exist. Reaching a global sample total alone does not override the existing
per-source and paired-training checks. This is an integrated budget gate and
finite executor, not a claim of validated 8-machine autonomous production.

## Reproducible commands

```sh
.venv/bin/python -m driversi.rsi.experiments.localization_free_retrieval \
  --project <repository> \
  --output <repository>/exp/chainflow9405-localization-free-v1

.venv/bin/python -m driversi.rsi.experiments.reactive_autoloop \
  --config exp/chainflow9405-localization-free-v1/wave-0001-config.json \
  --phase production
```

The first command refuses an existing completed output; reuse its queue rather
than repeat extraction. The second command was actually exercised and stopped
before generation with `production_budget_blocked / blocked_assets_or_repair`.
There was no new paid API call, navtrain student scoring, physics, render or
training in this run.

## Conditional timing

Using the previous service measurements, one successful generation attempt costs
roughly four to six serial minutes. Ten successful samples per source therefore
require about 40–60 minutes in the optimistic no-rejection case. For 562 sources,
that gives 374.7–562 serial hours. Ideal four/eight independent pipelines divide
this by four/eight; the current single-wave controller does not demonstrate that
throughput. These figures exclude reconstruction, retries, duplicate rejection,
queues, export and training. The prior 259-attempt batch produced only four
qualified 24 dB experimental samples and cannot establish the yield of this
stricter run.
