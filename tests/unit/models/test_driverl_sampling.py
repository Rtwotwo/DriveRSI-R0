import random

import numpy as np
import pytest

from driversi_agents.driverl import random_state


@pytest.mark.parametrize("method,search,parity", [("bogus", 0, False), ("sample", 8, False), ("sample", 0, True)])
def test_reject_ambiguous_native_generator(method, search, parity):
    with pytest.raises(ValueError):
        random_state.validate_sampling(method, search_candidates=search, parity=parity)


def test_rng_restore_replays_native_distribution_without_aliasing():
    torch = pytest.importorskip("torch")
    original = random_state.capture()
    try:
        torch.manual_seed(43)
        np.random.seed(43)
        random.seed(43)
        saved = random_state.capture()
        immutable = saved["torch"].clone()
        def draw():
            return (random.random(), float(np.random.random()), torch.distributions.Normal(torch.zeros(2), torch.ones(2)).sample())
        a = draw()
        draw()
        random_state.restore(saved)
        b = draw()
        assert a[:2] == b[:2]
        torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
        assert torch.equal(saved["torch"], immutable)
        torch.manual_seed(44)
        assert not torch.equal(draw()[2], a[2])
    finally:
        random_state.restore(original)


def test_native_sampling_family_keeps_generator_identity():
    from driversi.rsi.experiments.reactive_generation import assignment_plan
    from driversi.rsi.experiments.reactive_cohort import expand_plan
    row = dict(token="a", source_log="log", source_group="group")
    plan = assignment_plan([row], {"selection_role": "test"}, skill="driverl_sample_full_window")
    policy = dict(method="native_policy_sample_family_v1", attempts_per_anchor=10, seed_start=42)
    expanded = expand_plan(plan, policy)
    assert {a["sampling_seed"] for a in expanded["assignments"]} == set(range(42, 52))
    assert all(a["recipe"]["sampling_method"] == "sample" and a["recipe"]["search_candidates"] == 0 for a in expanded["assignments"])
    with pytest.raises(ValueError, match="skill differ"):
        expand_plan(assignment_plan([row], {"selection_role": "test"}, skill="driverl_full_window"), policy)
