import subprocess
import sys
import time

from driversi.runtime.experiment_guard import birth, settle


def test_native_child_stops_when_parent_exits(tmp_path):
    from driversi.runtime import parent_exec

    parent_script = tmp_path / "parent.py"
    pidfile, ready, late = (tmp_path / name for name in ("pid", "ready", "late"))
    child = (
        "from pathlib import Path; import time; Path(%r).write_text('ready'); time.sleep(10); Path(%r).write_text('bad')"
        % (str(ready), str(late))
    )
    parent_script.write_text(
        "import os,sys,subprocess,time\nfrom pathlib import Path\n"
        f"p=subprocess.Popen([sys.executable,{parent_exec.__file__!r},str(os.getpid()),sys.executable,'-c',{child!r}],start_new_session=True)\n"
        f"Path({str(pidfile)!r}).write_text(str(p.pid))\n"
        f"while not Path({str(ready)!r}).exists():time.sleep(.02)\n"
    )
    subprocess.run([sys.executable, str(parent_script)], check=True, timeout=5)
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 3
    while birth(pid) is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert birth(pid) is None
    assert not late.exists()


def test_guard_only_settles_its_feature_descriptor(tmp_path):
    import json

    script = tmp_path / "src/data/catalog/feature_worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("import time; time.sleep(60)")
    root = tmp_path / "owned"
    root.mkdir()
    config = root / "features.sqlite.feature-worker.json"
    config.write_text(json.dumps(dict(directory=str(root / "worker"))))
    foreign = tmp_path / "foreign.feature-worker.json"
    foreign.write_text(json.dumps(dict(directory=str(tmp_path / "foreign"))))
    processes = [
        subprocess.Popen([sys.executable, str(script), "--config", str(p)])
        for p in (config, foreign)
    ]
    try:
        rows = settle(root)
        assert len(rows) == 1 and rows[0]["pid"] == processes[0].pid and rows[0]["settled"]
        assert processes[1].poll() is None
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
            p.wait(timeout=3)
