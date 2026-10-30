import numpy as np
import pytest

from driversi.training.sampling.batch import learning_mask


def test_physical_validity_is_not_mutated_or_confused_with_learning():
    batch = dict(
        valid=np.array([True, True, False, True]),
        learner_mask=np.array([True, False, False, True]),
        behavior_policy_index=np.array([0, 1, -1, 0]),
        on_policy_eligible=np.array([True, True, False, False]),
    )
    np.testing.assert_array_equal(learning_mask(batch), [True, False, False, False])
    np.testing.assert_array_equal(batch["valid"], [True, True, False, True])


@pytest.mark.parametrize(
    "change",
    [
        {"learner_mask": np.array([1, 0])},
        {"on_policy_eligible": np.array([True])},
        {"behavior_policy_index": np.array([1, 0])},
        {"behavior_policy_index": np.array([0.0, 1.0])},
        {"behavior_policy_index": np.array([0, -1])},
    ],
)
def test_invalid_eligibility_rejected(change):
    batch = dict(
        valid=np.ones(2, bool),
        learner_mask=np.array([True, False]),
        on_policy_eligible=np.ones(2, bool),
        behavior_policy_index=np.array([0, 1]),
    )
    batch.update(change)
    with pytest.raises(ValueError):
        learning_mask(batch)


def test_missing_mixed_qualification_rejected():
    with pytest.raises(ValueError):
        learning_mask(dict(valid=np.ones(2, bool), behavior_policy_index=np.array([0, 1])))


def test_default_valid_is_preserved():
    value = np.array([True, False, True])
    np.testing.assert_array_equal(learning_mask({"valid": value}), value)
