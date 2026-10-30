#!/usr/bin/env python
"""Build a CPU-only navtrain history index from a manifest or streaming native logs."""

import argparse
import json
from pathlib import Path

from driversi.rsi.decisions.scene_index import build_index, describe_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--log-root", type=Path)
    parser.add_argument("--official-split", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-descriptors", type=Path)
    parser.add_argument("--development-group", action="append", default=[])
    parser.add_argument("--protected-group", action="append", default=[])
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    records, development = None, []
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        records = manifest["splits"]["train"]
        development = manifest["splits"]["development"]
    excluded = args.development_group + [row["source_log"] for row in development]
    result = build_index(
        records,
        official_split_path=args.official_split,
        development_groups=excluded,
        protected_groups=args.protected_group,
        log_root=args.log_root,
        max_scenes=args.max_scenes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.development_descriptors:
        args.development_descriptors.parent.mkdir(parents=True, exist_ok=True)
        args.development_descriptors.write_text(
            json.dumps(describe_records(development), indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps({"output": str(args.output), "coverage": result["coverage"]}))


if __name__ == "__main__":
    main()
