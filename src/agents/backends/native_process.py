"""Linux descendant ownership for native launchers (torchrun ranks create new sessions).

Use in a dedicated CLI/Harness worker with subreaper enabled. PID start times
prevent PID reuse from turning cleanup into a kill of an unrelated process.
"""

import os
import signal
import subprocess
import time
from pathlib import Path


def snapshot():
    values = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            s = (entry / "stat").read_text()
            fields = s[s.rfind(")") + 2 :].split()
            values[int(entry.name)] = {
                "parent": int(fields[1]),
                "start": fields[19],
                "state": fields[0],
            }
        except (OSError, ValueError, IndexError):
            continue
    return values


class OwnedTree:
    def __init__(self):
        self.owner = os.getpid()
        self.baseline = {
            (pid, row["start"]) for pid, row in snapshot().items() if row["parent"] == self.owner
        }
        self.owned = {}
        self.child = None

    def attach(self, child):
        self.child = child
        self.capture()

    def capture(self):
        rows = snapshot()
        if self.child is not None and self.child.pid in rows:
            self.owned.setdefault(self.child.pid, rows[self.child.pid]["start"])
        changed = True
        while changed:
            changed = False
            for pid, row in rows.items():
                identity = (pid, row["start"])
                parent = row["parent"]
                owned_parent = (
                    parent in self.owned
                    and parent in rows
                    and rows[parent]["start"] == self.owned[parent]
                )
                # Orphan rank grandchildren are adopted by this dedicated subreaper.
                adopted = parent == self.owner and identity not in self.baseline
                if pid not in self.owned and (owned_parent or adopted):
                    self.owned[pid] = row["start"]
                    changed = True
        return {pid: row for pid, row in rows.items() if self.owned.get(pid) == row["start"]}

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            self.capture()
            code = self.child.poll()
            if code is not None:
                return code
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.child.args, timeout)
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    def stop(self):
        if self.child is None:
            return
        # Signal every owned PID, including ranks in independent sessions. Keep
        # observing until descendants have exited and adopted zombies are reaped.
        for sig, grace in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 2.0)):
            deadline = time.monotonic() + grace
            while True:
                rows = self.capture()
                for pid, row in rows.items():
                    if row["state"] != "Z":
                        try:
                            # Recheck identity immediately before signalling.
                            text = (Path("/proc") / str(pid) / "stat").read_text()
                            start = text[text.rfind(")") + 2 :].split()[19]
                            if start == self.owned[pid]:
                                os.kill(pid, sig)
                        except (ProcessLookupError, FileNotFoundError):
                            pass
                self.child.poll()
                for pid, row in self.capture().items():
                    if pid != self.child.pid and row["parent"] == self.owner:
                        try:
                            os.waitpid(pid, os.WNOHANG)
                        except ChildProcessError:
                            pass
                if not self.capture():
                    time.sleep(0.02)
                    if not self.capture():
                        return
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.02)
        raise RuntimeError("Owned native descendants remain; resource release is unconfirmed")
