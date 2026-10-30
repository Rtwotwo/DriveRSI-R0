"""Compatibility helpers for NumPy pickles produced by newer NumPy releases."""

from __future__ import annotations

import sys

import numpy as np


_NUMPY_CORE_SUBMODULES = (
    "numeric",
    "multiarray",
    "umath",
    "_multiarray_umath",
    "_dtype",
    "_dtype_ctypes",
    "_internal",
    "_methods",
    "fromnumeric",
    "arrayprint",
    "records",
    "getlimits",
)


def install_numpy_2_pickle_compat() -> None:
    """Allow NumPy >=2 pickles to load in a NumPy <2 process."""
    try:
        __import__("numpy._core.numeric", fromlist=["numeric"])
        return
    except ImportError:
        pass

    import numpy.core as numpy_core

    # NumPy 1.26 exposes ``np._core`` on some builds even though imports such as
    # ``numpy._core.numeric`` do not resolve. Replace the incomplete alias.
    sys.modules["numpy._core"] = numpy_core
    for submodule in _NUMPY_CORE_SUBMODULES:
        try:
            module = __import__(f"numpy.core.{submodule}", fromlist=[submodule])
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{submodule}", module)
