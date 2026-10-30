"""Explicit, process-stable loading of a host-specific native backend."""

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import threading
from pathlib import Path
from types import ModuleType

_MODULE_NAME = "driversi.sim._native"
_LOCK = threading.Lock()


def load_native() -> ModuleType:
    """Load the default backend or the exact ``DRIVERSI_NATIVE_PATH`` file.

    Explicit selection is strict: an invalid path or ABI never falls back to
    the shared package extension. Changing the selection after loading is
    rejected; native worlds and snapshot ABIs are process-owned.
    """
    with _LOCK:
        selected = os.environ.get("DRIVERSI_NATIVE_PATH")
        cached = sys.modules.get(_MODULE_NAME)
        if selected is None:
            if cached is not None and getattr(cached, "_driversi_override_path", None):
                raise RuntimeError("Native backend selection changed; start a new process")
            return importlib.import_module(_MODULE_NAME)
        path = Path(selected).expanduser()
        if not selected or not path.is_absolute():
            raise ValueError("DRIVERSI_NATIVE_PATH must be an absolute extension file path")
        path = path.resolve(strict=True)
        if not path.is_file() or not any(
            path.name == "_native" + suffix
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        ):
            raise ValueError("DRIVERSI_NATIVE_PATH must name a compatible _native extension file")
        if cached is not None:
            if Path(cached.__file__).resolve() != path:
                raise RuntimeError("A different native backend is already loaded; start a new process")
            cached._driversi_override_path = str(path)
            return cached
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create native extension loader for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(_MODULE_NAME, None)
            raise
        module._driversi_override_path = str(path)
        setattr(importlib.import_module("driversi.sim"), "_native", module)
        return module
