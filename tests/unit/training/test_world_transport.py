import socket
import struct
import time
from multiprocessing.reduction import ForkingPickler

import numpy as np
import pytest

from driversi.training.sampling.world_transport import DeadlineChannel


def test_channel_roundtrip_preserves_numpy_and_message_identity():
    left, right = DeadlineChannel.pair()
    try:
        left.send((7, True, np.arange(32)), time.monotonic() + 1)
        sequence, ok, value = right.recv(time.monotonic() + 1)
        assert sequence == 7 and ok
        np.testing.assert_array_equal(value, np.arange(32))
    finally:
        left.close()
        right.close()


def test_nonreading_peer_cannot_block_large_send_past_deadline():
    left, right = DeadlineChannel.pair()
    try:
        left.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            left.send(bytes(2 ** 20), start + .05)
        assert time.monotonic() - start < 1
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("partial_header", [True, False])
def test_partial_response_is_also_deadline_bounded(partial_header):
    left, right = DeadlineChannel.pair()
    try:
        data = ForkingPickler.dumps((1, True, {"result": list(range(100))}))
        header = struct.pack("!Q", len(data))
        left.socket.sendall(header[:3] if partial_header else header + data[:3])
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            right.recv(start + .05)
        assert time.monotonic() - start < 1
    finally:
        left.close()
        right.close()


def test_peer_closing_mid_frame_raises_instead_of_returning_partial_data():
    left, right = DeadlineChannel.pair()
    left.socket.sendall(struct.pack("!Q", 200) + b"abc")
    left.close()
    try:
        with pytest.raises(EOFError, match="incomplete"):
            right.recv(time.monotonic() + 1)
    finally:
        right.close()
