import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build the stable DriveRSI visualization snapshot")
    parser.add_argument("command", choices=["build", "publish", "watch"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if args.command == "watch":
        from .watch import watch

        watch(args.config, args.interval)
        return
    from .build import build

    if args.command == "publish":
        from .publish import publish

        result = publish(args.config)
    else:
        result = build(args.config)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
