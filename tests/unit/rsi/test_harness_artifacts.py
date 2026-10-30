import copy

import pytest

from driversi.rsi.workflow.artifacts import seal, verify


def test_artifact_rejects_modified_file_and_foreign_branch(tmp_path):
    p = tmp_path / "data.npz"
    p.write_bytes(b"original")
    row = dict(id="dataset", output_kind="labels", provider="export", provider_version="1")
    result = dict(artifacts={"data": str(p)}, summary={})
    result["summary"]["artifact_reference"] = seal(result, tmp_path, "task", row)
    verify(result, "task", row)
    with pytest.raises(ValueError, match="Foreign"):
        verify(result, "other", row)
    p.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed"):
        verify(result, "task", row)


def test_artifact_rejects_bundle_substitution(tmp_path):
    p = tmp_path / "report.json"
    p.write_text("{}")
    row = dict(id="repair", output_kind="evidence", provider="repair", provider_version="1")
    result = dict(artifacts={"report": str(p)}, summary={})
    result["summary"]["artifact_reference"] = seal(result, tmp_path, "task", row)
    bad = copy.deepcopy(result)
    bad["artifacts"]["extra"] = str(p)
    with pytest.raises(ValueError, match="bundle"):
        verify(bad, "task", row)
    with pytest.raises(ValueError, match="outside"):
        seal(result, tmp_path / "child", "task", row)
