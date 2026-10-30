"""Build from clean source staging so moved modules cannot survive in build/lib."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def verify_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        layout = json.loads(archive.read("driversi/layout.json"))
        for old in layout["migrated"]:
            target = layout["module_moves"][old]
            canonical = target.replace(".", "/") + ".py"
            stale = old.replace(".", "/") + ".py"
            alias = "driversi/_compat/" + old.removeprefix("driversi.").replace(".", "/") + ".py"
            if canonical not in names or alias not in names or stale in names:
                raise RuntimeError(f"invalid migrated module in wheel: {old} -> {target}")


def build(output: Path) -> Path:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="driversi-wheel-") as temporary:
        stage = Path(temporary) / "source"
        stage.mkdir()
        for package in ("src", "agents"):
            shutil.copytree(
                ROOT / package,
                stage / package,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
            )
        for name in ("pyproject.toml", "README.md"):
            shutil.copy2(ROOT / name, stage / name)
        wheel_dir = Path(temporary) / "wheels"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheel_dir),
                str(stage),
            ],
            check=True,
        )
        wheels = list(wheel_dir.glob("driversi-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("expected one DriveRSI wheel")
        verify_contents(wheels[0])
        destination = output / wheels[0].name
        shutil.copy2(wheels[0], destination)
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
