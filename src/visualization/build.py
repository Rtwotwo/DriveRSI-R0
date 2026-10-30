"""Build a stable snapshot beside existing, explicitly published scene assets."""

import json
import os
from pathlib import Path

from .catalog import PROJECT, snapshot
from .architecture_figure import render_overview

ASSETS = Path(__file__).with_name("static")


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def build(config_path):
    config = json.loads(Path(config_path).read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported visualization configuration")
    output = Path(config["output"]).resolve()
    site = Path(config["site_root"]).resolve()
    if output != site / "platform":
        raise ValueError("Unified UI must be site_root/platform")
    from .current import current_config
    config = current_config(config)
    payload = snapshot(config)
    if payload.get("clustering"):
        from .clustering import public_clusters

        atomic(output / "clusters.json", json.dumps(public_clusters(config["clustering"], PROJECT), ensure_ascii=False, allow_nan=False))
    output.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "app.js", "current.js", "system.js", "clusters.js", "retrieval-comparison.js", "style.css", "whitebox-loader.js"):
        atomic(output / name, (ASSETS / name).read_text())
    if (payload.get("system") or {}).get("steps"):
        atomic(output / "architecture-overview.svg", render_overview(payload["system"]))
    source = (PROJECT / "src/rendering/inspection/showcase.html").read_text()
    source = source.replace("所有数据与页面保存于 32341 · DriveRSI/viz", "DriveRSI · 已发布的训练与工程案例")
    source = source.replace("<head>", '<head><base href="../">', 1)
    source = source.replace(
        "</style>",
        "body>header,.sidebar,.case-heading,.app>.main>.pillrow{display:none}.app{min-height:0;grid-template-columns:minmax(0,1fr)}.main{padding:10px}.workspace{grid-template-columns:minmax(280px,.95fr) minmax(340px,1.2fr)}body{background:#f3f6f9}</style>",
        1,
    )
    atomic(output / "scene.html", source)
    source = (PROJECT / "src/rendering/whitebox/viewer.html").read_text()
    source = source.replace(
        '<script src="geometry.js"></script><script src="viewer.js"></script>',
        '<script src="whitebox-loader.js"></script>',
    )
    source = source.replace(
        "</style>", "body>header{display:none}main{margin:0 auto;padding:0 8px}</style>", 1
    )
    atomic(output / "whitebox.html", source)
    for name in ("geometry.js", "viewer.js"):
        atomic(output / name, (PROJECT / "src/rendering/whitebox" / name).read_text())
    atomic(
        output / "platform.json", json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    )
    return {
        "status": "built",
        "output": str(output),
        "experiments": len(payload["experiments"]),
        "services": len(payload["services"]),
        "scenes": len(payload["scenes"]),
        "whitebox": len(payload["whitebox"]),
        "harness": len(payload["harness"]),
    }
