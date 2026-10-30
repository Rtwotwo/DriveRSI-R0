"""Bind a prebuilt gsplat module before its lazy loader invokes Ninja."""
import importlib.util
import os
import sys
from pathlib import Path


def load():
    root=os.environ.get('TORCH_EXTENSIONS_DIR')
    if not root:return None
    binary=Path(root)/'gsplat_cuda/gsplat_cuda.so'
    if not binary.is_file():return None
    if 'gsplat.cuda._backend' in sys.modules:
        bound=sys.modules.get('gsplat.csrc')
        if bound is not None and Path(bound.__file__).resolve()==binary.resolve():return binary.resolve()
        raise RuntimeError('A different gsplat backend was already loaded')
    import gsplat
    spec=importlib.util.spec_from_file_location('gsplat_cuda',str(binary))
    if spec is None or spec.loader is None:raise RuntimeError('Invalid prebuilt gsplat extension')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    gsplat.csrc=module
    sys.modules['gsplat.csrc']=module
    return binary.resolve()
