# Running and inspecting DriveRSI

[Installation](getting-started.md) · [Data](data-contract.md) · [Training and evaluation](train-eval.md) · [Workflow](workflow.md)

Run these commands from the checkout after activating the control-plane environment. Model workers have separate environments configured in their local plans. A help command inspects an interface; it does not prepare its datasets or prove that a GPU backend is ready.

## 1. Discover the installed platform

```bash
driversi --version
driversi agents
driversi services --group training
driversi config list --group rsi
python scripts/rsi/run_platform_spine.py inventory \
  --output exp/quickstart/interfaces.json
```

This writes an interface inventory without running models or contacting a Doctor. A registered service is an available adapter, not a completed experiment.

## 2. Run the CPU checks

```bash
python -m pytest -q \
  tests/unit/architecture/test_source_organization.py \
  tests/unit/rsi/test_harness_graph.py \
  tests/unit/rsi/test_dynamic_history_campaign.py \
  tests/unit/rendering/test_dynamic_history_binding.py \
  tests/unit/release/test_portable_release.py
```

These checks use fixtures and temporary outputs. The broader release suite and exact counts are recorded in [release verification](release-verification.md). Native GPU tests need their own dependencies and real inputs.

## 3. Prepare sources and a generation plan

```bash
python scripts/data/run_scene_corpus.py --help
python scripts/rsi/run_synthesis_campaign.py --help
python scripts/rsi/run_platform_spine.py continue --help
```

Choose source scenes from the training pool, keep the test failure separately, and declare image horizons, teacher runtime, assets, concurrency, per-source attempts, timeout and total budget. Store the plan under a new `exp/<run-id>/`. Existing corpus, generation, and task formats are indexed in [the code map](code-map.md).

For a **prepared and reviewed** synthesis configuration, the execution entry is:

```bash
python scripts/rsi/run_synthesis_campaign.py \
  --config /absolute/path/to/prepared-campaign.json --phase production
```

This command executes production work. It is not a dry run. A generic machine-independent campaign cannot substitute for source-specific maps, native environments, weights, and evidence paths.

## 4. Reuse physical branches for dynamic rendering and replay

Inspect the exact plan validator and help before creating a new plan:

```bash
python -m driversi.rsi.experiments.dynamic_history_campaign --help
```

The campaign expects sealed physical rollouts and diversity reports, reconstruction jobs, source identities, resource limits, and a native replay request. It renders the student's required historical/anchor images, then checks binding and sample admission. It does not create missing teacher rollouts or relax failed quality gates.

```bash
# Runs GPU work after all required inputs have been prepared.
python -m driversi.rsi.experiments.dynamic_history_campaign \
  --plan /absolute/path/to/prepared-dynamic-plan.json \
  --output exp/my-dynamic-replay
```

The current inherited experiment has unresolved binding issues described in [release status](release-status.md); this command is not a promise that arbitrary reconstructed assets will pass.

## 5. Control a registered Harness task

```bash
python -m driversi.rsi.workflow.engine --help
python -m driversi.rsi.workflow.engine --registry exp/my-run/harness.sqlite status TASK_ID
python -m driversi.rsi.workflow.engine --registry exp/my-run/harness.sqlite events TASK_ID
python -m driversi.rsi.workflow.engine --registry exp/my-run/harness.sqlite cancel TASK_ID
python -m driversi.rsi.workflow.engine --registry exp/my-run/harness.sqlite recover TASK_ID
```

Replace `TASK_ID` with the existing task identifier. `cancel` requests stopping owned work; `recover` reconciles owner and child-process evidence. It does not unconditionally replay work. Preserve spent budgets and completed results. An explicit pause takes precedence over automatic progression.

## Local visualization

```bash
python -m driversi.visualization build \
  --config src/config/visualization/platform.json
python -m http.server 8000 --bind 127.0.0.1 --directory exp/local-view
```

Open `http://127.0.0.1:8000/platform/`. The default is an empty inspection platform. Create a local config copy and register only the experiment receipts and scene media you intend to inspect. Rebuild after changing those registrations.

The server binds to loopback and serves only the generated inspection directory. There is no GitHub Pages workflow, upload, background monitor, or implicit training launch.

## Reading progress

Count samples only after all required admission stages complete. Keep attempted, teacher-passed, distinct, rendered, and admitted counts separate. Record both summed task time and wall time. Inspect process ownership and settled output files alongside progress JSON; a stale PID or progress label is not evidence that work is still running.
