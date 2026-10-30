# Installation and configuration

[Run and inspect](running.md) · [Training and evaluation](train-eval.md) · [Data contracts](data-contract.md)

## 1. Install the control plane

Use Linux, Python 3.11+, a compiler for optional simulator builds, and sufficient local scratch space. From the source checkout:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
driversi --version
driversi agents
driversi services
```

Core dependencies are NumPy and PyYAML. `dev` adds pytest and ruff; `clustering` adds scikit-learn/SciPy; `maps` adds the map-processing packages. These extras are defined in [pyproject.toml](../pyproject.toml). Use editable source installation: native workers and vendor discovery refer to the checkout, so a wheel alone is not a complete GPU platform deployment.

## 2. Prepare data and initial models

Obtain NAVSIM/OpenScene/nuPlan data and maps under their original terms. Follow the upstream [NAVSIM installation guide](https://github.com/autonomousvision/navsim/blob/main/docs/install.md). Keep original data separately from generated data:

```text
data/             original datasets, maps and native metric caches
weights/          downloaded parent models and backbones
dataset/          newly generated packages (sensor_blobs/, navsim_logs/)
lib/              optional catalog, retrieval, caption and cluster libraries
exp/<run-id>/     per-run inputs, receipts, budgets, checkpoints and reports
```

These paths are conventions, not downloads. This release includes none of the original datasets, previously generated samples, trained assets, credentials, caches or experiment checkpoints. Keep all original split and source-group identifiers so train/test isolation can be checked.

## 3. Configure your machine explicitly

Presets in `src/config/` are templates. Absolute external paths starting with `/path/to/` must be replaced with actual inputs. A value containing `@DRIVERSI_ROOT@` resolves to the current source checkout through `driversi.configuration.read_mapping` and `driversi.config.load`; it is not shell or Hydra interpolation.

To make an explicit copy for a tool that reads plain JSON directly:

```bash
python scripts/build/materialize_config.py \
  --input src/config/agents/chainflow_vla/lifecycle.yaml \
  --output src/config/local/chainflow.json
```

Edit the generated file to set interpreter, weights, maps, data and permitted GPU resources. The command does not start a job and does not overwrite existing output. `src/config/local/` is ignored by Git. Native Hydra configurations (such as the EPDMS scorer YAML) must remain in their native loader; do not pass them through the plain-mapping materializer.

Model launch configurations usually resolve relative paths against the repository root. For campaign inputs requiring absolute paths, supply resolved paths explicitly. Do not transfer old sealed receipts by search-and-replacing their paths; prepare a new run and record the evidence it reuses.

## 4. Install native environments separately

| Backend | Starting point | Important distinction |
|---|---|---|
| AUGUstus | [Official installation instructions](https://github.com/haha-yuki-haha/AUGUstus/blob/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_SFT/docs/INSTALL.md), [bundled implementation](../agents/augustus/native/) | Checkpoint, processor, Torch, CUDA and optional attention kernels must match |
| ChainFlow | [Official installation guide](https://github.com/AFARI-Research/ChainFlow-VLA/blob/main/docs/install.md), [bundled native source](../agents/chainflow/native/), [lifecycle configuration](../src/config/agents/chainflow_vla/lifecycle.yaml) | Use the bundled revision's stage1 model, backbone and native preprocessing; upstream may have changed |
| DriveRL/DriveZero | [Bundled source](../agents/driverl/native/README.md) | Teacher execution and the platform's learning implementation are distinct |
| MTGS | [Bundled source](../src/vendor/rendering/mtgs/README.md) | Reconstruction, rendering and CUDA-extension validation have separate dependencies |
| NAVSIM | [v1 source](../src/vendor/driving/navsim-v1.1/README.md), [v2 source](../src/vendor/driving/navsim-v2.2/README.md) | Select the scoring protocol; do not mix metric caches across versions |

Set the resulting interpreter paths in local configurations. Do not copy a compiled `gsplat_cuda.so` from another machine: validate the native build and its ABI in the environment that will run it. The platform retains a witness mechanism for known builds, not a portable precompiled binary.

For the optional simulator extension, inspect `python scripts/build/build_pufferdrive.py --help`; this is separate from control-plane installation. GPU resource allocation is explicit. Source code does not imply that a particular GPU or another user's running process is available.

## 5. Optional Doctor / caption service

Paid or private model services are not required to inspect the source, run the CPU checks, or use deterministic rendering/qualification code. The existing private-network client requires an explicit deployed model configuration and an owner-only credential file; it has no automatic public-provider fallback.

Set `DRIVERSI_PRIVATE_ROOT` to a private directory outside the checkout (default: `~/.config/driversi/private`). Keep credential and private endpoint configuration files under this directory with mode `0600`. Pass file paths through the campaign's documented fields. No key values belong in presets, traces, README or Git history. Adaptive handoff requires the private configuration to be supplied in the existing generation policy; it no longer assumes a particular server's credential location.

## 6. Local inspection

```bash
python -m driversi.visualization build --config src/config/visualization/platform.json
python -m http.server 8000 --bind 127.0.0.1 --directory exp/local-view
```

The default configuration has no previous experiment registrations. Serve only the generated inspection directory, not the repository or data root. No service is started by installation and no Pages workflow is generated.
