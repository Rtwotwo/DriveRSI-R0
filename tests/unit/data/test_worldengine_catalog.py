import pytest
from driversi.data.assets.worldengine_catalog import ROOT, collect_pages

def entry(i):
    return dict(Path=f"{ROOT}/part{i:03d}.tar.gz", Revision="frozen-revision", Size=100)

def response(total, ids):
    return {"Data": {"TotalCount": total, "Files": [entry(i) for i in ids]}}

def test_all_129_archives_require_second_page():
    calls = []
    def fetch(page, size):
        calls.append(page)
        return response(129, range(1,101) if page == 1 else range(101,130))
    result = collect_pages(fetch)
    assert calls == [1,2]
    assert len(result["files"]) == 129 and result["complete"]
    assert result["trainable"] is False

@pytest.mark.parametrize("second,match", [
    (response(3, [2]), "Repeated"),
    (response(4, [3]), "changed"),
    (response(3, []), "Incomplete"),
])
def test_inconsistent_pagination_cannot_be_called_complete(second, match):
    with pytest.raises(ValueError, match=match):
        collect_pages(lambda p,s: response(3,[1,2]) if p == 1 else second, page_size=2)

def test_page_budget_does_not_return_partial_success():
    with pytest.raises(ValueError, match="page budget"):
        collect_pages(lambda p,s:response(3,[1,2]),page_size=2,max_pages=1)
