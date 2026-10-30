import pytest

from driversi.rendering.gaussian.source_preflight import CAMERAS, source_window


def frames():
    return [dict(token=str(i), timestamp=i*500000,
                 cams={c: {} for c in CAMERAS}) for i in range(20)]


def test_exact_anchor_and_past_future_window():
    selected = source_window(frames(), "6")
    assert [f["token"] for f in selected] == list(map(str, range(2, 17)))


def test_dense_metadata_uses_capture_times_not_row_offsets():
    rows = [dict(token=str(i), timestamp=i*100000,
                 cams={c: {} for c in CAMERAS}) for i in range(100)]
    assert [f["token"] for f in source_window(rows, "30")] == list(map(str, range(10, 81, 5)))


@pytest.mark.parametrize("fault", ("gap", "missing_camera", "duplicate", "boundary"))
def test_incomplete_or_ambiguous_assets_reject(fault):
    rows = frames()
    token = "6"
    if fault == "gap":
        rows[9]["timestamp"] += 100000
    elif fault == "missing_camera":
        rows[4]["cams"].pop("CAM_B0")
    elif fault == "duplicate":
        rows[7]["token"] = token
    else:
        token = "3"
    with pytest.raises(ValueError):
        source_window(rows, token)
