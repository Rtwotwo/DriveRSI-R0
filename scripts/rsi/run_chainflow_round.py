"""Compile an admitted ChainFlow data increment; --execute runs the Harness."""

import argparse
import json

from driversi.rsi.experiments.chainflow_round import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run(args.config, execute=args.execute)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] in ("prepared", "feedback_available") else 2)
