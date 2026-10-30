"""The reorganization must retain live old callers without duplicate implementations."""

import importlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from driversi.architecture import inventory, structure_violations

ROOT = Path(__file__).resolve().parents[3]


def test_all_implementations_have_functional_ownership():
    report = inventory()
    assert report["implemented_modules"] == len(report["module_moves"])
    assert not report["missing_implementations"]
    assert structure_violations() == []


@pytest.mark.parametrize(
    "old",
    [
        "driversi.rsi.harness",
        "driversi.rsi.harness_training",
        "driversi.rsi.budget",
        "driversi.training.engine",
        "driversi.training.online",
        "driversi.training.environment",
        "driversi.agents.native",
        "driversi.data.nuplan_db",
        "driversi.sim.pufferdrive",
        "driversi.rendering.replay",
        "driversi.evaluation.nuplan_metrics",
    ],
)
def test_old_import_is_same_module_and_monkeypatch_target(old, monkeypatch):
    target = inventory()["module_moves"][old]
    a = importlib.import_module(old)
    b = importlib.import_module(target)
    assert a is b and Path(a.__file__).resolve() == Path(b.__file__).resolve()
    monkeypatch.setattr(a, "_layout_identity_fixture", object(), raising=False)
    assert a._layout_identity_fixture is b._layout_identity_fixture


def test_old_serialized_class_reference_resolves_to_same_class():
    from driversi.training.sampling.environment import TeacherEnvironment

    assert (
        pickle.loads(b"cdriversi.training.environment\nTeacherEnvironment\n.") is TeacherEnvironment
    )


def test_old_and_new_module_commands_have_same_effect(tmp_path):
    results = []
    for module in ["driversi.training.engine", "driversi.training.orchestration.engine"]:
        output = tmp_path / (module + ".json")
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                module,
                "inspect",
                str(ROOT / "src/config/training/driverl.json"),
                "--output",
                str(output),
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(output.read_text()))
    assert results[0] == results[1]


def test_canonical_build_entry_runs_outside_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build/build_pufferdrive.py"), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "--source" in result.stdout


def test_old_budget_module_cli_stays_lazy_without_torch(tmp_path):
    code = "import sys; import driversi.rsi.budget, driversi.training.job; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, check=True)


@pytest.mark.parametrize(
    "script",
    ["diagnostics/export_driverl_failure_analysis.py",
     "legacy/reports/export_driverl_navigation_repair.py"],
)
def test_report_scripts_resolve_shared_code_without_running_main(script, tmp_path):
    path = ROOT / "scripts" / script
    code = (
        "import runpy, sys; from pathlib import Path; "
        "p=Path(sys.argv[1]); sys.path.insert(0,str(p.parent)); "
        "namespace=runpy.run_path(str(p),run_name='import_only'); "
        "from driversi.rendering.inspection.driverl import overview; "
        "assert namespace['overview'] is overview"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], cwd=tmp_path, check=True)
