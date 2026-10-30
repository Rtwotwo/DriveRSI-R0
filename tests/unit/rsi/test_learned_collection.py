import pytest

from driversi.agents.backends.native import digest
from driversi.rsi.plugins.collection import validate_student


def receipt(tmp_path):
    p = tmp_path / "weights.pt"
    p.write_bytes(b"candidate")
    student = {
        "status": "completed",
        "model": "driverl",
        "phase": "resume",
        "context": "recipe",
        "checkpoint": str(p),
        "checkpoint_files": {str(p): digest(p)},
        "source_groups": {"train": ["train"], "development": ["heldout"]},
    }
    plan = {"model_config": {"model": "driverl"}, "context": "recipe"}
    return student, plan


def test_candidate_does_not_require_false_promotion(tmp_path):
    student, plan = receipt(tmp_path)
    student["promoted"] = False
    validate_student(student, plan, "train")


def test_never_recycle_development(tmp_path):
    student, plan = receipt(tmp_path)
    with pytest.raises(ValueError, match="Development"):
        validate_student(student, plan, "heldout")


def test_other_model_or_unpublished_checkpoint_rejected(tmp_path):
    student, plan = receipt(tmp_path)
    student["model"] = "augustus"
    with pytest.raises(ValueError, match="DriveRL"):
        validate_student(student, plan, "train")
    student["model"] = "driverl"
    student["checkpoint"] = str(tmp_path / "other.pt")
    with pytest.raises(ValueError, match="publication"):
        validate_student(student, plan, "train")


def test_mutated_checkpoint_rejected(tmp_path):
    student, plan = receipt(tmp_path)
    from pathlib import Path

    Path(student["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        validate_student(student, plan, "train")
