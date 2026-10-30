"""Exec a native command only while its explicit spawning parent is alive."""

import ctypes
import os
import signal
import sys


def main(argv):
    parent = int(argv[0])
    if len(argv) < 2 or os.getppid() != parent:
        raise RuntimeError("Native launcher lost its parent before binding")
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0 or os.getppid() != parent:
        raise RuntimeError("Cannot bind native command lifetime")
    os.execvpe(argv[1], argv[1:], os.environ)


if __name__ == "__main__":
    main(sys.argv[1:])
