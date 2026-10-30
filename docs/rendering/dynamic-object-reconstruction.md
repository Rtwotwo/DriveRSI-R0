# Measured dynamic object reconstruction

`driversi.rendering.gaussian.dynamic_reconstruction` wraps the pinned MTGS ST
trainer, without changing its Gaussian models, losses or 100-point eligibility.
Unlike the older background-only entry point, it sets `only_moving=False`, so
measured parked vehicles are represented by actual token-bound rigid nodes too.
Background, sky and rigid vehicles are optimized together in a new asset bundle.
Existing background assets and their receipts are never overwritten.

Preparation reuses the source image, semantic mask, LiDAR registration, COLMAP
calibration and depth products. Object initialization uses all recorded LiDAR
sensors rather than only the top sensor, upstream object-local box transforms,
and colors sampled from the calibrated source images. No invented points or
painted boxes are substituted for missing objects. Exact repeated returns are
deduplicated before the native point-count test. Pedestrians/bicycles are not
supported by this rigid adapter and remain explicit in the coverage manifest.

```python
job = prepare_dynamic_job(recovery_job, output, gpu=0, max_iterations=30001)
```

Run with the existing reconstruction environment:

```
python -m driversi.rendering.gaussian.dynamic_reconstruction --job /absolute/job.json
```

`report.json` returns `completed`, `bundle_root`, `checkpoint`, `metadata`,
`binding_manifest`, `object_tokens`, and `unsupported_objects`. Completion means
the native bundle has finite, nonempty rigid nodes matching measured eligibility;
it does not grant training admission. `source_quality_pending=true` requires a
fresh source-view PSNR check. The renderer must separately prove same-branch
object placement and reject unsupported visible actors. Source PSNR is not novel
view pixel ground truth.

The bundle contains `dynamic-binding-manifest.json` with checkpoint, metadata and
point-inventory hashes, token→node mapping and native Gaussian counts. Source
files and pinned producer code are hashed in the job. Completed reruns verify
hashes and return existing results. Interrupted training can resume only native
checkpoints with optimizer state; the controller owns finite retry limits.

The preprocessing point cloud is measured support, not a claim of full visual
coverage. Sparse distant vehicles can remain unsupported after all-sensor
accumulation; their visibility must be checked rather than silently waived.

## Verified native CUDA runtime

`driversi.rendering.gaussian.verified_gsplat_runtime` can reuse the already built
official `gsplat_cuda.so` without launching Ninja in every trainer. It checks the
extension SHA256, producer SHA256, Python version, Torch/CUDA version, gsplat
version and C++ ABI against a saved real forward/backward witness. It does not
implement a replacement rasterizer. A missing or mismatched witness fails closed.

```python
from driversi.configuration import read_mapping
from driversi.rendering.gaussian.verified_gsplat_runtime import abi, validate_witness, load_extension
load_extension(validate_witness(read_mapping(witness_path), abi()))
```

For an existing immutable reconstruction job, use:

```
python -m driversi.rendering.gaussian.verified_gsplat_runtime --job /absolute/job.json --witness /absolute/witness.json
```

The wrapper preserves the native job and child stages, propagating the explicit
witness to subprocesses. It does not launch or retry assets independently of
Harness. The recorded runtime witness exercises actual spherical harmonics,
rasterization and backward gradients with JIT disabled; it is runtime evidence,
not an asset-quality or data-admission result.
