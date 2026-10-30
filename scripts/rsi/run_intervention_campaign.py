"""Prepare or advance a bounded native paired-training intervention campaign."""

import argparse
import json

from driversi.rsi.experiments.intervention_campaign import advance, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "advance"))
    parser.add_argument("path")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.operation == "prepare":
        if args.execute:
            parser.error("prepare never executes training")
        result = prepare(args.path)
        print(json.dumps({"campaign_id": result["campaign_id"], "prepared": True}))
    else:
        print(json.dumps(advance(args.path, execute=args.execute)))


if __name__ == "__main__":
    main()
