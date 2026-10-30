import numpy as np
import pytest

from driversi.data.scenes.pufferdrive import write_binary

native = pytest.importorskip("driversi.sim._native")
pytestmark = pytest.mark.native


def test_binary_load_joint_action_and_replay_velocity(straight_scene, tmp_path):
    path = tmp_path / "scene.bin"
    write_binary(straight_scene, path)
    handle = native.create(str(path))
    try:
        before = native.read(handle)
        np.testing.assert_array_equal(before[2], [0, 1])
        np.testing.assert_allclose(before[1][:, 4], 4)
        native.step(handle, np.asarray([7, 10], dtype=np.int32))
        obs, state, _, _, step = native.read(handle)
        assert step == 1
        assert obs.shape == (2, native.observation_size)
        assert np.isfinite(obs).all()
        assert state[0, 0] == pytest.approx(0.4, abs=1e-4)
        assert state[1, 0] > state[0, 0]
        assert state[2, 0] == pytest.approx(0.4, abs=1e-4)
        assert state[2, 4] == pytest.approx(4)
    finally:
        native.close(handle)
        native.close(handle)
