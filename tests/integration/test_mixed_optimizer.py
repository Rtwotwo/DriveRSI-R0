# ruff: noqa: E402
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from driversi.training.legacy.checkpoint import restore_checkpoint, save_checkpoint
from driversi.training.ppo import update


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(4, 3)
        self.critic = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.actor(x), self.critic(x).squeeze(-1)


def test_opponent_and_intervened_rows_never_enter_optimizer():
    torch.manual_seed(7)
    model = Model()
    reference = copy.deepcopy(model)
    rng = np.random.default_rng(8)
    data = dict(
        observations=rng.normal(size=(6, 4)).astype(np.float32),
        actions=np.array([0, 1, 2, 0, 1, 2]),
        log_probs=np.full(6, -1.1, np.float32),
        advantages=rng.normal(size=6).astype(np.float32),
        returns=np.ones(6, np.float32),
        values=np.zeros(6, np.float32),
        valid=np.ones(6, bool),
    )
    selected = np.array([True, False, True, False, False, True])
    clean = {k: v[selected] for k, v in data.items()}
    data.update(
        learner_mask=np.array([True, False, True, True, False, True]),
        behavior_policy_index=np.array([0, 1, 0, 0, 1, 0]),
        on_policy_eligible=np.array([True, True, True, False, True, True]),
    )
    # Poison non-learning rows: a faulty implementation would immediately fail.
    data["observations"][~selected] = np.nan
    state = torch.get_rng_state()
    update(model, torch.optim.Adam(model.parameters()), data, epochs=2, minibatch=2)
    torch.set_rng_state(state)
    update(reference, torch.optim.Adam(reference.parameters()), clean, epochs=2, minibatch=2)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, reference.state_dict()[k], rtol=0, atol=0)


def test_skipped_batch_counter_survives_checkpoint(tmp_path):
    class Sampler:
        def snapshot(self):
            return {"step": 9}

        def restore(self, state):
            assert state == {"step": 9}

    model = Model()
    opt = torch.optim.Adam(model.parameters())
    sampler = Sampler()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, opt, sampler, 4, 123, {}, optimizer_updates=2)
    assert restore_checkpoint(path, model, opt, sampler, {}, return_counters=True) == (4, 123, 2)
    assert restore_checkpoint(path, model, opt, sampler, {}) == (4, 123)
