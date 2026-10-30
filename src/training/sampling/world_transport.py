"""Framed local RPC with one deadline across all send/receive I/O.

Only trusted child processes exchange these pickles. Socket handles support spawn;
no background thread can remain blocked after a pool closes its owned children.
Serialization is local CPU work, not preempted by the I/O deadline.
"""
import socket
import struct
import time
from multiprocessing.reduction import ForkingPickler


class DeadlineChannel:
    def __init__(self, sock):
        self.socket = sock

    @classmethod
    def pair(cls):
        left, right = socket.socketpair()
        return cls(left), cls(right)

    def _timeout(self, deadline):
        if deadline is None:
            self.socket.settimeout(None)
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("World worker RPC deadline exceeded")
            self.socket.settimeout(remaining)

    def send(self, value, deadline=None):
        data = ForkingPickler.dumps(value)
        self._timeout(deadline)
        self.socket.sendall(struct.pack("!Q", len(data)))
        self._timeout(deadline)
        self.socket.sendall(data)

    def _read(self, size, deadline):
        data = bytearray(size)
        view = memoryview(data)
        offset = 0
        while offset < size:
            self._timeout(deadline)
            count = self.socket.recv_into(view[offset:])
            if not count:
                raise EOFError("World worker closed an incomplete RPC frame")
            offset += count
        return data

    def recv(self, deadline=None):
        size, = struct.unpack("!Q", self._read(8, deadline))
        return ForkingPickler.loads(self._read(size, deadline))

    def close(self):
        self.socket.close()
