import numpy as np
import pytest

torch = pytest.importorskip("torch")
from driversi_agents.driverl.learning.learner import (  # noqa: E402 - optional Torch gate
    LearningConfig,
    decomposed_gae,
    restore_rng,
    rng_state,
)


def test_terminal_blocks_bootstrap_and_truncation_keeps_it():
    rewards = torch.tensor([[1.0, 2.0]])
    values = torch.tensor([[0.1, 0.2]])
    bootstrap = torch.tensor([10.0, 20.0])
    adv, returns = decomposed_gae(
        rewards, values, bootstrap, torch.tensor([True]), torch.tensor([False]), 0.9, 0.8
    )
    torch.testing.assert_close(returns, rewards)
    _, returns = decomposed_gae(
        rewards, values, bootstrap, torch.tensor([False]), torch.tensor([True]), 0.9, 0.8
    )
    torch.testing.assert_close(returns, torch.tensor([[10.0, 20.0]]))


def test_component_advantages_conserve_scalar_gae():
    r = torch.arange(27, dtype=torch.float32).reshape(3, 9) / 100
    v = torch.ones_like(r) / 3
    b = torch.ones(9)
    terminal = torch.tensor([False, False, False])
    truncated = torch.tensor([False, False, True])
    a, ret = decomposed_gae(r, v, b, terminal, truncated)
    scalar, _ = decomposed_gae(
        r.sum(-1, keepdim=True), v.sum(-1, keepdim=True), b.sum().view(1), terminal, truncated
    )
    torch.testing.assert_close(a.sum(-1), scalar[:, 0])


def test_reset_inside_batch_is_rejected():
    with pytest.raises(ValueError, match="reset"):
        decomposed_gae(
            torch.zeros(2, 9),
            torch.zeros(2, 9),
            torch.zeros(9),
            torch.tensor([True, False]),
            torch.tensor([False, False]),
        )


def test_rng_restores_native_stochastic_actions():
    state = rng_state()
    expected = torch.distributions.Beta(torch.ones(2) * 2, torch.ones(2) * 3).sample((8,))
    restore_rng(state)
    actual = torch.distributions.Beta(torch.ones(2) * 2, torch.ones(2) * 3).sample((8,))
    assert torch.equal(expected, actual)


@pytest.mark.parametrize(
    "kwargs",
    [{"learning_rate": float("nan")}, {"minibatch_size": 0}, {"epochs": True}, {"gamma": 2}],
)
def test_bad_training_config(kwargs):
    with pytest.raises(ValueError):
        LearningConfig(**kwargs).validate()


def test_reward_memory_survives_departure_and_remaps_baselines():
    from types import SimpleNamespace

    from driversi_agents.driverl.learning.reward_memory import RewardMemory

    memory = RewardMemory()
    lines = np.array([[[0.0, 0.0], [1.0, 0.0]], [[2.0, 0.0], [3.0, 0.0]]])
    data = SimpleNamespace(
        npc_mask_all=torch.ones(1, 3, 3, dtype=torch.bool),
        agent_goal_reached_all=torch.zeros(1, 3, 3, dtype=torch.bool),
        _nuplan_ttc_triggered=torch.tensor([[True, False, False]]),
        _nuplan_ttc_collided_matrix=torch.tensor(
            [[[False, True, False], [False] * 3, [False] * 3]]
        ),
        _direction_baseline_last=torch.tensor([[0, 0, 0]]),
        _direction_progress_last=torch.tensor([[0.7, 0.0, 0.0]]),
    )
    memory.capture(data, ["ego", "left", "right"], lines, True)
    other = SimpleNamespace(
        npc_mask_all=torch.ones(1, 3, 3, dtype=torch.bool),
        agent_goal_reached_all=torch.zeros(1, 3, 3, dtype=torch.bool),
    )
    memory.apply(other, ["ego", "new", "right"], lines[::-1].copy())
    assert other._nuplan_ttc_triggered[0, 0] and other.agent_goal_reached_all[0, 0, -1]
    assert other._direction_baseline_last[0, 0] == 1
    assert not other._nuplan_ttc_collided_matrix.any()
    memory.apply(other, ["ego", "new", "left"], lines[::-1].copy())
    assert other._nuplan_ttc_collided_matrix[0, 0, 2]
    memory.apply(other, ["ego", "new", "left"], lines + 10)
    assert other._direction_baseline_last[0, 0] == -1


@pytest.mark.parametrize("native_agrees", [True, False])
def test_world_collision_stops_first_substep_and_is_never_ignored(native_agrees):
    from types import SimpleNamespace

    from driversi_agents.driverl.learning.environment import NativeEnvironment

    calls = []
    live = [True]
    world = SimpleNamespace(
        episode_done=False, observe=lambda: SimpleNamespace(alive=[live[0]], step=len(calls))
    )

    def step(**kwargs):
        calls.append(kwargs)
        live[0] = False
        return SimpleNamespace(
            events={"collision": [True]},
            terminated=[True],
            truncated=[False],
            observation=world.observe(),
        )

    env = NativeEnvironment.__new__(NativeEnvironment)
    env.session = SimpleNamespace(
        world=world,
        step=step,
        history=[{}],
        controller=SimpleNamespace(get_state=lambda: "pre_stop"),
    )
    env.ended = False
    env.history = []
    env.steps = 0
    env.physics_steps = 0
    env.horizon = 110
    env.world_events = []
    env.observation = lambda: {"reward_components": np.array([-1.0] + [0.0] * 8)}

    def reward(ego):
        assert ego == "pre_stop"
        env.last_reward_evidence = {"native_collision": native_agrees, "native_offroad": False}
        return (-1.0 if native_agrees else 0.0), native_agrees

    env._reward = reward
    result = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert len(calls) == 1 and env.ended
    assert result.platform_valid == native_agrees
    if native_agrees:
        assert result.reward == -1.0 and result.terminated and not result.truncated
    else:
        assert result.truncated and result.intervention == "world_native_terminal_disagreement"
    with pytest.raises(RuntimeError, match="ended"):
        env.step(np.zeros(2, dtype=np.float32))


def test_recovery_rejects_partially_updated_learner_even_with_same_version(tmp_path):
    from types import SimpleNamespace

    from driversi_agents.driverl.learning.checkpoint import save

    actor = torch.nn.Linear(2, 2)
    learner = torch.nn.Linear(2, 2)
    learner.load_state_dict(actor.state_dict())
    with torch.no_grad():
        learner.weight.add_(0.1)
    collector = SimpleNamespace(
        unresolved_step=False, version="v", actor=SimpleNamespace(policy_version="v", agent=actor)
    )
    with pytest.raises(ValueError, match="weights are not synchronized"):
        save(
            tmp_path / "bad.pt",
            collector,
            SimpleNamespace(parent_version="v", agent=learner),
            seal={},
        )
    assert not (tmp_path / "bad.pt").exists()
