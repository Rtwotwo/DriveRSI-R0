import json
from pathlib import Path

import pytest

from driversi.agents.backends.native import digest
from driversi.rsi.services.native_export_worker import NativeExportSession


def test_unknown_export_never_launches(tmp_path):
    d = tmp_path / "job"
    d.mkdir()
    (d / "launch.json").write_text("{}")
    session = NativeExportSession("/unavailable/python", 1, tmp_path / "service")
    with pytest.raises(ValueError, match="Unsettled"):
        session("chainflow_export", {"plan": {}}, d, gpu=1, python="/unavailable/python")
    assert session.process is None


def test_completed_export_checks_result_hash(tmp_path):
    d = tmp_path / "job"
    d.mkdir()
    (d / "job.json").write_text(json.dumps(dict(action="chainflow_export", arguments={"plan": {}})))
    (d / "result.json").write_text('{"done":true}')
    (d / "exit.json").write_text(
        json.dumps(
            dict(
                returncode=0,
                job_sha256=digest(d / "job.json"),
                result_sha256=digest(d / "result.json"),
            )
        )
    )
    session = NativeExportSession("/unavailable/python", 1, tmp_path / "service")
    session("chainflow_export", {"plan": {}}, d, gpu=1, python="/unavailable/python")
    assert session.process is None
    (d / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="result changed"):
        session("chainflow_export", {"plan": {}}, d, gpu=1, python="/unavailable/python")


def test_worker_identity_does_not_migrate_gpu(tmp_path):
    session = NativeExportSession("/unavailable/python", 1, tmp_path)
    with pytest.raises(ValueError, match="GPU changed"):
        session(
            "chainflow_export", {"plan": {}}, tmp_path / "job", gpu=2, python="/unavailable/python"
        )


def test_venv_interpreter_entry_preserves_environment(tmp_path):
    import subprocess
    import venv

    root = tmp_path / "isolated_env"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    entry = root / "bin/python"
    session = NativeExportSession(entry, 1, tmp_path / "worker")
    result = subprocess.run(
        [session.python, "-c", "import sys;print(sys.prefix)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(session.python) == entry
    assert Path(result.stdout.strip()) == root
