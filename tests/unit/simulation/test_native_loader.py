"""Host selection must fail closed instead of silently using another ABI."""

import importlib.machinery
import sys
from types import ModuleType, SimpleNamespace

import pytest

from driversi.sim.backends import native_loader

NAME = "driversi.sim._native"


@pytest.fixture
def native_selection(monkeypatch, tmp_path):
    path = tmp_path / ("_native" + importlib.machinery.EXTENSION_SUFFIXES[0])
    path.write_bytes(b"test loader fixture, not an executable")
    monkeypatch.setenv("DRIVERSI_NATIVE_PATH", str(path))
    monkeypatch.delitem(sys.modules, NAME, raising=False)
    return path


@pytest.mark.parametrize("value", ["", "relative/_native.so"])
def test_override_requires_absolute_path(monkeypatch, value):
    monkeypatch.setenv("DRIVERSI_NATIVE_PATH", value)
    with pytest.raises(ValueError, match="absolute"):
        native_loader.load_native()


def test_missing_override_never_falls_back(native_selection, monkeypatch):
    native_selection.unlink()
    monkeypatch.setattr(native_loader.importlib, "import_module",
                        lambda *_: pytest.fail("Default import must not run"))
    with pytest.raises(FileNotFoundError):
        native_loader.load_native()


def test_override_rejects_non_extension(native_selection, monkeypatch):
    path = native_selection.with_name("_native.py")
    path.write_text("raise AssertionError('must not execute')")
    monkeypatch.setenv("DRIVERSI_NATIVE_PATH", str(path))
    with pytest.raises(ValueError, match="extension"):
        native_loader.load_native()


def test_loaded_backend_cannot_be_replaced(native_selection, monkeypatch):
    existing = ModuleType(NAME)
    existing.__file__ = str(native_selection.with_name("other.so"))
    monkeypatch.setitem(sys.modules, NAME, existing)
    with pytest.raises(RuntimeError, match="different native backend"):
        native_loader.load_native()
    assert sys.modules[NAME] is existing


def test_same_backend_reuses_module_and_removing_override_is_rejected(native_selection, monkeypatch):
    existing = ModuleType(NAME)
    existing.__file__ = str(native_selection)
    monkeypatch.setitem(sys.modules, NAME, existing)
    assert native_loader.load_native() is existing
    monkeypatch.delenv("DRIVERSI_NATIVE_PATH")
    with pytest.raises(RuntimeError, match="selection changed"):
        native_loader.load_native()


def test_failed_explicit_module_cleans_import_cache(native_selection, monkeypatch):
    module = ModuleType(NAME)

    def reject(_):
        raise ImportError("unsupported host ABI")

    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=reject))
    monkeypatch.setattr(native_loader.importlib.util, "spec_from_file_location", lambda *_: spec)
    monkeypatch.setattr(native_loader.importlib.util, "module_from_spec", lambda *_: module)
    with pytest.raises(ImportError, match="unsupported host ABI"):
        native_loader.load_native()
    assert NAME not in sys.modules


def test_successful_explicit_load_publishes_exact_module(native_selection, monkeypatch):
    import driversi.sim

    monkeypatch.setattr(driversi.sim, "_native", None, raising=False)
    module = ModuleType(NAME)
    module.__file__ = str(native_selection)
    calls = []
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda value: calls.append(value)))
    monkeypatch.setattr(native_loader.importlib.util, "spec_from_file_location", lambda *_: spec)
    monkeypatch.setattr(native_loader.importlib.util, "module_from_spec", lambda *_: module)
    assert native_loader.load_native() is module
    assert native_loader.load_native() is module
    assert calls == [module]
    assert driversi.sim._native is module


def test_unset_override_preserves_default_import(monkeypatch):
    monkeypatch.delenv("DRIVERSI_NATIVE_PATH", raising=False)
    monkeypatch.delitem(sys.modules, NAME, raising=False)
    expected = ModuleType(NAME)
    calls = []

    def default_import(name):
        calls.append(name)
        return expected

    monkeypatch.setattr(native_loader.importlib, "import_module", default_import)
    assert native_loader.load_native() is expected
    assert calls == [NAME]
