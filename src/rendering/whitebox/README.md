# Whitebox rendering

Maintained implementation for structured source-log and executed-branch inspection, calibrated semantic video conditions, and verified delivery.

| Module | Responsibility |
|---|---|
| `conditioning.py` | Source measurements, native camera mounts, continuous resampling, executed-branch provenance and structured export |
| `projection_profiles.py` | Explicit display-only projection policies; preserves stored measurements |
| `render_conditions.py` | Four-camera rasterization and video/frame receipts |
| `motion_audit.py` | Independent source-motion and calibration fidelity checks; never certifies a physical branch |
| `delivery.py` | Bounded batch execution, immutable reuse checks and verified download packaging |
| `delivery_prompt.py`, `minimax_h3.py` | Versioned generation prompts and H3 request assembly; no generation/admission claim |
| `exporter.py`, `schema.py`, `viewer.*`, `geometry.js` | Structured log/branch replay and schema validation |

Source imports use `driversi.rendering.whitebox.*`; no implementation is imported from `exp/`, `outputs/` or `agents/`.

The native source coordinate convention and the display projection are separate. `planar_consistent_v1` aligns road, box bottoms and camera in a planar view; it does not establish terrain height, real occlusion, GS alignment or training eligibility. Executed branches retain recorded physics states and explicit camera mounts. Historical exports keep their original convention; current batch delivery rejects unrepaired geometry and requires a new output directory.

Canonical entries from the repository root:

```bash
.venv/bin/python scripts/data/export_semantic_conditions.py --help
.venv/bin/python scripts/data/run_whitebox_delivery.py --help
.venv/bin/python scripts/rendering/export_whitebox.py --help
```

Reusable presets live in `src/config/rendering/`. Regression tests live in `tests/unit/rendering/`; generated JSON, frames, videos, logs and audit reports live in `exp/<run-id>/`. Experiment-specific reproduction scripts are retained only as run evidence, not runtime dependencies.

See [delivery protocol](../../../docs/data/whitebox-delivery.md) and [repository organization](../../../docs/architecture/repository-layout.md).

Source videos now declare `source_bracket_linear_v1`: annotation box dimensions use the same timestamp bracket as centers and headings. Exact source observations and missing-identity holds remain unchanged. This removes artificial shape steps, not annotation noise; physical branches retain executed dimensions. Current delivery requires a new export for legacy stepped-size bundles. The independent raw-log audit recognizes both policies without rewriting old evidence.
