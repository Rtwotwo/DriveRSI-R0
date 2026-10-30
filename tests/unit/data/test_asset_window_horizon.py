import runpy
from pathlib import Path
from driversi.agents.backends.native import project_root

module=runpy.run_path(str(project_root()/'scripts/data/audit_navsim_asset_windows.py'))

def test_chainflow_cannot_accept_only_eight_future_points():
    starts=module['complete_window_starts']
    assert starts(range(13),future_points=8)==[0]
    assert starts(range(13),future_points=10)==[]
    assert starts(range(15),future_points=10)==[0]


def test_full_span_does_not_hide_internal_missing_frame():
    starts=module['complete_window_starts']
    assert starts([i for i in range(15)if i!=13])==[]
    assert starts(range(20))==list(range(6))
