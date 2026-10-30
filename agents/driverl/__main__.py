"""Run with python -m driversi_agents.driverl --config <trusted YAML>."""
import argparse
import json

from driversi_agents.driverl.runtime import inspect_runtime, run_preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-preflight", action="store_true",
                        help="Execute official read-only CPU preflight in the configured environment")
    args = parser.parse_args(argv)
    try:
        report = inspect_runtime(args.config)
        if args.run_preflight:
            report = run_preflight(report)
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc), "trainable": False}))
        return 2
    print(json.dumps(report, indent=2))
    if args.run_preflight:
        return 0 if report["official_preflight"]["status"] == "passed" else 2
    return 0 if report["can_attempt_official_preflight"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
