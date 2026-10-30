"""Installed entry points must not depend on the checkout working directory."""
import json
import subprocess
import sys
from pathlib import Path


def test_installed_cli_and_agent_discovery_from_other_directory(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-m", "driversi", "legacy-agents"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    rows = json.loads(result.stdout)["agents"]
    assert rows == []
    assert all(Path(row["path"]).is_file() for row in rows)


def test_installed_namespace_uses_flat_source_without_exposing_top_level_agents(tmp_path):
    code = (
        "from pathlib import Path; import driversi, driversi.agents; "
        "root=Path(driversi.__file__).resolve().parent; "
        "assert root.name == 'src'; "
        "assert Path(driversi.agents.__file__).resolve().parent == root/'agents'; "
        "assert not (root/'driversi').exists()"
    )
    subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, check=True)
