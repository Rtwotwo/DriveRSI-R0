"""Native snapshots must not encode allocator residue for replay actors."""

import json
import os
import subprocess
import sys

import pytest

from driversi.data.scenes.pufferdrive import write_binary

pytest.importorskip("driversi.sim._native")
pytestmark = pytest.mark.native

# Fresh glibc processes make missing scalar/padding initialization reproducible,
# instead of relying on the incidental contents of a reused allocation.
_PROBE = r"""
import hashlib
import json
import sys
import numpy as np
from driversi.sim.backends.native_loader import load_native

native = load_native()
records = []
for reset in range(2):
    handle = native.create(sys.argv[1])
    try:
        for tick in range(11):
            if tick:
                native.step(handle, np.asarray([10, 7], dtype=np.int32))
            if tick in (0, 10):
                obs, state, indices, rewards, step = native.read(handle)
                assert np.isfinite(obs).all() and np.isfinite(state).all()
                assert indices.tolist() == [0, 1]
                records.append({
                    "reset": reset, "tick": step,
                    "snapshot": native.snapshot(handle).hex(),
                    "observable_sha256": hashlib.sha256(
                        obs.tobytes() + state.tobytes() + rewards.tobytes()
                    ).hexdigest(),
                })
    finally:
        native.close(handle)
print(json.dumps(records))
"""


def test_mixed_actor_snapshots_do_not_depend_on_allocator_residue(straight_scene, tmp_path):
    assert straight_scene.controlled.tolist() == [True, True, False]
    straight_scene.poses[-1, :2, 0] = 100
    binary = tmp_path / "mixed-scene.bin"
    write_binary(straight_scene, binary)
    results = []
    for perturb in (85, 170):
        env = dict(os.environ, MALLOC_PERTURB_=str(perturb),
                   GLIBC_TUNABLES="glibc.malloc.tcache_count=0", PYTHONDONTWRITEBYTECODE="1")
        process = subprocess.run(
            [sys.executable, "-B", "-c", _PROBE, str(binary)], env=env,
            check=True, text=True, capture_output=True, timeout=60,
        )
        results.append(json.loads(process.stdout))
    # Dynamics matching alone does not excuse non-deterministic hidden state.
    assert [x["observable_sha256"] for x in results[0]] == [
        x["observable_sha256"] for x in results[1]
    ]
    assert results[0] == results[1]
    for run in results:
        assert run[0]["snapshot"] == run[2]["snapshot"]
        assert run[1]["snapshot"] == run[3]["snapshot"]
