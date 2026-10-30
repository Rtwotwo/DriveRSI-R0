"""Linux-owned tool process groups; trusted plugins must not detach from the group."""
import ctypes
import os
import signal
import subprocess
import time
from pathlib import Path


def subreaper(enable=None):
    """Return prior setting; adopt orphan helpers so their usage can be reaped."""
    libc = ctypes.CDLL(None, use_errno=True)
    prior = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(prior), 0, 0, 0):
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER")
    if enable is not None and libc.prctl(36, int(enable), 0, 0, 0):
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")
    return bool(prior.value)


def group_members(pgid):
    members = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text()
            parts = text[text.rfind(")") + 2:].split()
            if int(parts[2]) != pgid:
                continue
            members[int(entry.name)] = {
                "parent": int(parts[1]), "alive": parts[0] != "Z", "start": parts[19],
                "cpu": sum(int(parts[i]) for i in (11, 12, 13, 14)) / os.sysconf("SC_CLK_TCK"),
            }
        except (OSError, ValueError, IndexError):
            continue
    return members


def reap_helpers(child):
    # Popen owns the leader's wait status. Only reap adopted helpers in this group.
    for pid, value in group_members(child.pid).items():
        if pid != child.pid and value["parent"] == os.getpid():
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass


def stop_group(child):
    if child is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=.3)
        except subprocess.TimeoutExpired:
            pass
        if sig == signal.SIGKILL:
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                child.poll()
                reap_helpers(child)
                if not group_members(child.pid):
                    return
                time.sleep(.02)
    raise RuntimeError("Owned tool group has not exited; reservation retained for explicit recovery")
