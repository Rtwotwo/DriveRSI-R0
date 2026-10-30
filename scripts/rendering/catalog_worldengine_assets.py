"""Read all WorldEngine archive metadata pages without downloading assets."""
import argparse
import json
from pathlib import Path
from driversi.data.assets.worldengine_catalog import fetch_catalog

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=20)
    args = parser.parse_args()
    result = fetch_catalog(max_pages=args.max_pages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "files"}))
