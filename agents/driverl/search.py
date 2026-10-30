"""Explicit correction window; not an implicit replacement of the baseline."""


def search_window(candidates=0, start=None, stop=None, *, seed=42):
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Native TTS seed must be an integer in [0, 2^31)")
    if (
        isinstance(candidates, bool)
        or not isinstance(candidates, int)
        or candidates not in (0, 8, 64)
    ):
        raise ValueError("Search candidates must be 0, 8 or 64")
    if candidates == 0:
        if start is not None or stop is not None:
            raise ValueError("Disabled search cannot have a window")
        return None
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, stop)):
        raise ValueError("Search window requires integer ticks")
    if not 0 <= start < stop <= 6000:
        raise ValueError("Search window outside bounded rollout")
    return dict(
        candidates=candidates,
        start_tick=start,
        stop_tick=stop,
        seed=seed,
        policy="official_native_TTS",
        baseline_comparable=False,
    )


def repair_window(backend=None, start=None, stop=None):
    """A distinct correction policy; never advertised as native model parity."""
    if backend is None:
        if start is not None or stop is not None:
            raise ValueError("Disabled repair cannot have a window")
        return None
    if backend != "pdm_route_v1":
        raise ValueError("Unknown correction backend")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, stop)):
        raise ValueError("Repair window requires integer ticks")
    if not 0 <= start < stop <= 6000:
        raise ValueError("Repair window outside bounded rollout")
    return dict(
        backend=backend,
        start_tick=start,
        stop_tick=stop,
        baseline_comparable=False,
        native_model_modified=False,
        planner="NAVSIM-v1.1-PDMClosed",
        tracker="nuPlan-LQR-bounded",
        future_observations=False,
        training=False,
    )
