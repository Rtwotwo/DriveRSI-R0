# Release verification

Date: 2026-10-30. Source baseline: `2725bed62f70e3fefa6b3880a341412425cbce8b`, followed by the release changes listed in `release-manifest.json`.

## Verified scope

| Check | Observed result |
|---|---|
| Clean Python 3.11 editable installation with `.[dev]` | Passed |
| Isolated import origin | Both `driversi` and `driversi_agents` resolved to the release checkout |
| CLI version, agent/service/config discovery | Passed |
| Platform interface inventory | Passed; 94 registered providers; no claim of full RSI validation |
| Empty local visualization build | Passed; zero registered experiments and scenes |
| Selected clean-environment CPU regression suite | **191 passed, 0 failed, 0 skipped**, 8.74 seconds |
| Maintained Python source syntax | 1,284 files parsed; no syntax errors |
| Project YAML presets | 14 parsed; no parse errors |
| Root-token configuration and materialization | Covered by regression tests |
| Missing private configuration at handoff | Rejected before consuming the handoff; covered by regression tests |

The test count is one clean-environment run, not the sum of repeated runs. The earlier 178-test check was preparatory. Dependency downloads used the Aliyun PyPI mirror after the default route was slow; this is package installation, not a model API call.

## Reproduce the regression subset

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit/architecture/test_source_organization.py \
  tests/unit/rsi/test_harness_graph.py \
  tests/unit/rsi/test_harness_contracts.py \
  tests/unit/rsi/test_harness_artifacts.py \
  tests/unit/rsi/test_dynamic_history_campaign.py \
  tests/unit/rendering/test_dynamic_history_binding.py \
  tests/unit/rsi/test_rsi_promotion.py \
  tests/unit/rsi/test_platform_spine.py \
  tests/unit/visualization/test_platform.py \
  tests/unit/rendering/test_showcase_pages.py \
  tests/unit/release/test_portable_release.py
```

These tests cover source organization, task graph/contracts, artifacts, admission, decisions, and local export. They use fixtures; no native GPU training, teacher rollout, reconstruction, full benchmark, or paid model request was executed as part of packaging. The original experiment remains paused.

## Payload provenance

The release retains native/vendor source manifests and licenses. The overview PNG and PDF are unchanged copies of the manuscript figure; exact hashes are in [paper-figure-provenance.json](assets/paper-figure-provenance.json). They are architecture illustrations, not new benchmark evidence.

The payload excludes original datasets, generated training samples, checkpoints, credentials, historical experiment outputs, Git history, and public site payloads. Empty data-layout documentation is included. `release-manifest.json` and `release-files.sha256` record the final source payload and checksums.

Known experimental limitations are described separately in [release status](release-status.md).
