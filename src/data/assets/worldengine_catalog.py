"""Complete, bounded metadata inventory; downloading assets is a separate action."""
import json
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = "data/sim_engine/assets/navtrain/assets"
ENDPOINT = "https://www.modelscope.cn/api/v1/datasets/OpenDriveLab/WorldEngine/repo/tree"

def collect_pages(fetch, *, page_size=100, max_pages=20):
    if type(page_size) is not int or not 1 <= page_size <= 100 or type(max_pages) is not int or max_pages < 1:
        raise ValueError("Positive bounded pagination required")
    rows, seen, receipts, total = [], set(), [], None
    for page in range(1, max_pages + 1):
        response = fetch(page, page_size)
        data = response["Data"]
        count, files = data["TotalCount"], data["Files"]
        if type(count) is not int or count < 0 or not isinstance(files, list) or len(files) > page_size:
            raise ValueError("Malformed asset inventory page")
        if total is not None and count != total:
            raise ValueError("Asset inventory changed during pagination")
        total = count
        if not files and len(rows) != total:
            raise ValueError("Incomplete asset inventory: premature empty page")
        for item in files:
            path = item.get("Path", "")
            if not path.startswith(ROOT + "/") or "/" in path[len(ROOT)+1:] or not path.endswith(".tar.gz"):
                raise ValueError("Unexpected asset archive path")
            if path in seen:
                raise ValueError("Repeated asset path across inventory pages")
            if not isinstance(item.get("Revision"), str) or not item["Revision"] or type(item.get("Size")) is not int or item["Size"] <= 0:
                raise ValueError("Versioned nonempty archive metadata required")
            seen.add(path)
            rows.append(item)
        receipts.append(dict(page=page, received=len(files), total=count))
        if len(rows) > total:
            raise ValueError("Inventory exceeds declared total")
        if len(rows) == total:
            return dict(schema_version=1, kind="worldengine_complete_archive_inventory", root=ROOT,
                        total_count=total, complete=True, pages=receipts, files=rows,
                        trainable=False, scope="archive inventory only; no asset or scene qualification")
    raise ValueError("Asset inventory exceeds page budget; not complete")

def fetch_catalog(*, timeout=30, max_pages=20):
    def fetch(page, size):
        query = urlencode(dict(Revision="master", Root=ROOT, PageNumber=page, PageSize=size))
        with urlopen(ENDPOINT + "?" + query, timeout=timeout) as response:
            return json.load(response)
    return collect_pages(fetch, max_pages=max_pages)
