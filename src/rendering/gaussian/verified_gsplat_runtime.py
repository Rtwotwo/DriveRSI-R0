"""Load an ABI-pinned official gsplat binary without repeating its JIT build.

This only selects a previously built native module. Gaussian algorithms, native
training configuration, immutable asset jobs and their receipts are untouched.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json

MODULE = 'driversi.rendering.gaussian.verified_gsplat_runtime'
WITNESS_ENV = 'DRIVERSI_VERIFIED_GSPLAT_RUNTIME'


def abi():
    import torch
    import gsplat
    return {'python': '.'.join(map(str, sys.version_info[:2])), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gsplat': gsplat.__version__,
            'cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI)}


def validate_witness(value, current_abi):
    if (value.get('schema_version') != 1 or value.get('kind') != 'verified_official_gsplat_runtime'
            or value.get('checks_passed') is not True or value.get('abi') != current_abi
            or value.get('producer_sha256') != digest(__file__)):
        raise ValueError('Verified matching gsplat runtime witness required')
    path = Path(value['extension']).resolve(strict=True)
    allowed = (project_root() / '.runtimes').resolve()
    if not path.is_relative_to(allowed) or path.name != 'gsplat_cuda.so':
        raise ValueError('Official extension must belong to the existing project runtime')
    if digest(path) != value['extension_sha256']:
        raise ValueError('Verified official extension changed')
    return path


def load_extension(path):
    """Pybind's real init name remains gsplat_cuda; expose official csrc alias."""
    import torch  # Loads the matching libtorch dependencies before dlopen.
    import gsplat
    path = Path(path).resolve(strict=True)
    existing = sys.modules.get('gsplat_cuda')
    if existing is not None and Path(existing.__file__).resolve() != path:
        raise ValueError('Another gsplat native module is already loaded')
    if existing is None:
        spec = importlib.util.spec_from_file_location('gsplat_cuda', path)
        existing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(existing)
        sys.modules['gsplat_cuda'] = existing
    gsplat.csrc = existing
    sys.modules['gsplat.csrc'] = existing
    from gsplat.cuda import _backend
    if _backend._C is not existing:
        raise ValueError('gsplat backend did not bind the verified native module')
    return existing


def verify(extension, output, gpu):
    output = Path(output).resolve()
    if not output.is_relative_to((project_root() / 'exp').resolve()):
        raise ValueError('Owned experiment witness output required')
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import time
    start = time.monotonic()
    current = abi()
    path = Path(extension).resolve(strict=True)
    candidate = {'schema_version': 1, 'kind': 'verified_official_gsplat_runtime',
                 'checks_passed': True, 'abi': current, 'producer_sha256': digest(__file__),
                 'extension': str(path), 'extension_sha256': digest(path)}
    validate_witness(candidate, current)
    import torch
    import torch.utils.cpp_extension as cpp
    # Verification fails if anything tries to invoke a compiler, even when an
    # apparently successful image might otherwise hide a fallback JIT path.
    original = cpp._run_ninja_build
    def forbidden(*args, **kwargs):
        raise RuntimeError('Unexpected JIT invocation in verified binary path')
    cpp._run_ninja_build = forbidden
    try:
        load_extension(path)
        torch.manual_seed(42)
        torch.cuda.set_per_process_memory_fraction(.05)
        from gsplat.rendering import rasterization
        from gsplat.cuda._wrapper import spherical_harmonics
        means = torch.tensor([[0., 0., 3.], [.2, .1, 4.]], device='cuda', requires_grad=True)
        quats = torch.tensor([[1., 0., 0., 0.]] * 2, device='cuda')
        scale = torch.full((2, 3), .1, device='cuda')
        opacity = torch.full((2,), .8, device='cuda')
        colors = torch.rand((2, 3), device='cuda')
        view = torch.eye(4, device='cuda')[None]
        intr = torch.tensor([[[32., 0., 16.], [0., 32., 16.], [0., 0., 1.]]], device='cuda')
        image, alpha, _ = rasterization(means, quats, scale, opacity, colors, view, intr, 32, 32, packed=False)
        image.sum().backward()
        assert torch.isfinite(image).all() and means.grad is not None and torch.isfinite(means.grad).all()
        assert float(means.grad.abs().sum()) > 0
        coeff = torch.rand((2, 4, 3), device='cuda', requires_grad=True)
        directions = torch.tensor([[0., 0., 1.], [.2, .1, .9]], device='cuda')
        sh = spherical_harmonics(1, directions, coeff)
        sh.sum().backward()
        assert torch.isfinite(sh).all() and coeff.grad is not None and torch.isfinite(coeff.grad).all()
        assert float(coeff.grad.abs().sum()) > 0
        torch.cuda.synchronize()
        candidate.update(device_capability=list(torch.cuda.get_device_capability()),
                         raster_sum=float(image.sum()), gradient_sum=float(means.grad.sum()),
                         sh_gradient_abs_sum=float(coeff.grad.abs().sum()), jit_invoked=False,
                         elapsed_seconds=time.monotonic() - start, trainable=False)
    finally:
        cpp._run_ninja_build = original
    if output.exists() and read_mapping(output) != candidate:
        raise ValueError('Preserve old runtime witness; choose a fresh path')
    write_json(output, candidate)
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job')
    parser.add_argument('--witness')
    parser.add_argument('--stage', choices=['train'])
    parser.add_argument('--verify-extension')
    parser.add_argument('--output')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.verify_extension:
        result = verify(args.verify_extension, args.output, args.gpu)
        print(json.dumps({k: result[k] for k in ('checks_passed', 'abi', 'elapsed_seconds', 'jit_invoked')}))
        return
    witness_path = args.witness or os.environ.get(WITNESS_ENV)
    if not witness_path:
        raise ValueError('Explicit verified runtime witness is required')
    witness_path = str(Path(witness_path).resolve(strict=True))
    value = read_mapping(witness_path)
    path = validate_witness(value, abi())
    load_extension(path)
    os.environ[WITNESS_ENV] = witness_path
    from driversi.rendering.gaussian import dynamic_reconstruction as native
    # The native wrapper already delegates training to a child interpreter.
    # Make that child use this same verified-loader entrypoint; no native source
    # or immutable job is patched and no training setting is changed.
    native.MODULE = MODULE
    if args.stage:
        job = read_mapping(args.job)
        native._verify(job)
        native._train(job)
    else:
        report = native.run_dynamic_job(args.job)
        print(json.dumps({key: report[key] for key in ('status', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
