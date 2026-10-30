"""Real orphan helper cleanup within the isolated Harness plugin worker."""
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process ownership")
@pytest.mark.parametrize("worker_exit", [0, 7])
def test_plugin_worker_reaps_detached_orphan_without_parent_subreaper(tmp_path, worker_exit):
    pids = tmp_path / "pids.json"
    native = tmp_path / "native.py"
    native.write_text(
        "import subprocess,sys,time,json\n"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)\n"
        f"open({str(pids)!r},'w').write(json.dumps([p.pid]))\n"
        "time.sleep(.2)\n"
        f"sys.exit({worker_exit})\n"
    )
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "import sys,json\nfrom pathlib import Path\n"
        "from driversi.runtime.processes import subreaper\n"
        "from driversi.rsi.plugins.synthesis import run_process\n"
        "subreaper(False)\n"
        "try:\n"
        f" run_process([sys.executable,{str(native)!r}],Path({str(tmp_path)!r}),10)\n"
        "except RuntimeError as exc:\n"
        f" assert {worker_exit} != 0 and 'Skill worker failed' in str(exc), str(exc)\n"
        "else:\n"
        f" assert {worker_exit} == 0\n"
        "assert subreaper() is False\n"
        f"assert not any(Path('/proc/'+str(pid)).exists() for pid in json.loads(Path({str(pids)!r}).read_text()))\n"
    )
    completed = subprocess.run([sys.executable, str(wrapper)], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
