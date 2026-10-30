"""Install the stable local site entry without Git or network operations."""

import json
from pathlib import Path

from .build import atomic, build

MARKER = "driversi-stable-entry-v1"


def publish(config_path):
    result = build(config_path)
    config = json.loads(Path(config_path).read_text())
    site = Path(config["site_root"]).resolve()
    entry, legacy = site / "index.html", site / "legacy-showcase.html"
    if entry.exists() and MARKER not in entry.read_text():
        if legacy.exists() and legacy.read_bytes() != entry.read_bytes():
            raise ValueError(
                "Legacy landing already differs; refusing to overwrite historical entry"
            )
        if not legacy.exists():
            atomic(legacy, entry.read_text())
    atomic(
        entry,
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="driversi-stable-entry-v1" content="1"><title>DriveRSI</title>
<p><a href="platform/index.html">打开 DriveRSI 统一工作台</a></p>
<script>
const query=new URLSearchParams(location.search);const target=new URL('platform/index.html',location.href);
const key=query.get('case');if(key){const route=new URLSearchParams({view:'scenes',case:key,mode:'gs'});for(const name of ['tick','camera'])if(query.has(name))route.set(name,query.get(name));target.hash=route.toString();}
location.replace(target.href);
</script></html>""",
    )
    result["landing"] = str(entry)
    result["git_pushed"] = False
    return result
