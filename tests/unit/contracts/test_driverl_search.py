import pytest

from driversi_agents.driverl.search import search_window


def test_baseline_and_explicit_window():
    assert search_window() is None
    result = search_window(8, 260, 310)
    assert result["baseline_comparable"] is False
    assert result["start_tick"] == 260 and result["stop_tick"] == 310


@pytest.mark.parametrize(
    "args",
    [(True, 0, 10), (16, 0, 10), (0, 0, 10), (8, None, 10), (8, 2, 2), (8, -1, 10), (64, 0, 6001)],
)
def test_invalid_window(args):
    with pytest.raises(ValueError):
        search_window(*args)
