"""Isolated image-environment worker for independent video quality diagnostics."""

import argparse
import json
import sys
import types
from pathlib import Path

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[3]
    for name, folder in (("driversi", "src"), ("driversi_agents", "agents")):
        module = types.ModuleType(name)
        module.__path__ = [str(root / folder)]
        sys.modules[name] = module

from driversi.rendering.video.validation import source_comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("novel", "source", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    report = source_comparison(args.novel, args.source)
    Path(args.output).write_text(json.dumps(report, indent=2))
