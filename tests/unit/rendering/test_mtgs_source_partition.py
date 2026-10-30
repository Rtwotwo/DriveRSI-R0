from types import SimpleNamespace as NS
import pytest
from driversi.rendering.gaussian.mtgs_runtime import select_source_dataset


def test_existing_native_eval_cameras_are_accessible_without_model_change():
    train = NS(_dataparser_outputs=NS(travel_ids=[0, 1, 7]))
    evaluate = NS(_dataparser_outputs=NS(travel_ids=[0, 1, 6, 7]))
    manager = NS(train_dataset=train, eval_dataset=evaluate)
    assert select_source_dataset(manager) == (train, "train")
    assert select_source_dataset(manager, 0) == (train, "train")
    assert select_source_dataset(manager, 6) == (evaluate, "eval")
    with pytest.raises(ValueError, match="no native source camera"):
        select_source_dataset(manager, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        select_source_dataset(manager, True)
