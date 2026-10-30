# Code map

| Responsibility | Maintained implementation | Notes |
|---|---|---|
| Public contracts | `src/contracts/` | Scenes, actors, observations and model capability |
| Data catalog and semantic libraries | `src/data/catalog/` | Caption, retrieval and clustering code; no paid calls by installation |
| Source logs and maps | `src/data/sources/`, `src/data/road_network/` | NAVSIM/nuPlan identity and geometry |
| Training data export | `src/data/datasets/` | Native packages, history/anchor image contracts |
| Physical execution | `src/sim/` | Joint world, lifecycle, persistence and workers |
| Gaussian reconstruction and rendering | `src/rendering/gaussian/` | MTGS adapters, dynamic reconstruction and source comparison |
| Image and branch qualification | `src/rendering/diagnostics/` | Numerical/geometry checks, dynamic binding |
| DriveHarness execution | `src/rsi/workflow/`, `src/rsi/execution/` | DAG, typed stages, registry, owned processes and budgets |
| Skill decisions and experience | `src/rsi/decisions/`, `src/rsi/correction/` | Retrieval decisions, admissibility, proposal filtering and memory |
| Native campaigns | `src/rsi/experiments/` | Continual curriculum, paired training and frozen-skill experiments |
| Service adapters | `src/rsi/plugins/`, `src/rsi/services/` | Concrete providers, not a second execution engine |
| Training lifecycle | `src/training/` | Explicit native backends, data exposure and resume |
| Native model implementations | `agents/` | Separate upstream source, adapters and environments |
| Evaluation | `src/evaluation/` | Diagnosis, metrics, official scoring and EPDMS |
| Local inspection | `src/visualization/` | Read-only views and allowlisted evidence projection |
| Upstream dependencies | `src/vendor/` | Pinned original namespaces and notices |

Thin launchers live in `scripts/<domain>/`. [The command catalog](../scripts/catalog.json) names their implementation and lifecycle status. Historical report commands remain under `scripts/legacy/` for inspecting old evidence; they are not default experiment launchers.

`src/layout.json` records earlier module moves required by serialized Python objects. Do not delete these compatibility identities just to shorten the tree. Optional vendor code is isolated in workers rather than added globally to `PYTHONPATH`.

See [repository layout](architecture/repository-layout.md) and [source organization](architecture/source-organization.md) for the detailed ownership rules. The root package maps `driversi` to `src` and `driversi_agents` to `agents` in `pyproject.toml`; folder names are intentional.
