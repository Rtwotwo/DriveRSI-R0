"""Register and execute bounded development experiments; keep browser endpoints read-only."""
import argparse
import json
from pathlib import Path

from driversi.configuration import read_mapping
from driversi.rsi.execution.registry import ExperimentRegistry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("list")
    teacher = sub.add_parser("admit-teacher-search")
    teacher.add_argument("file", type=Path)
    teacher.add_argument("--output", type=Path, required=True)
    for name in ("register", "record"):
        sub.add_parser(name).add_argument("file", type=Path)
    run = sub.add_parser("execute")
    run.add_argument("file", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    for name in ("status", "cancel", "recover"):
        sub.add_parser(name).add_argument("job_id")
    campaign_register = sub.add_parser("campaign-register")
    campaign_register.add_argument("file", type=Path)
    campaign_register.add_argument("--output-root", type=Path, required=True)
    for name in ("campaign-run", "campaign-status", "campaign-cancel", "campaign-recover"):
        sub.add_parser(name).add_argument("campaign_id")
    repair_gate = sub.add_parser("repair-gate")
    repair_gate.add_argument("file", type=Path)
    repair_gate.add_argument("--output", type=Path, required=True)
    repair = sub.add_parser("repair-run")
    repair.add_argument("file", type=Path)
    repair.add_argument("--output", type=Path, required=True)
    for name in ("repair-gate-status", "repair-gate-recover"):
        sub.add_parser(name).add_argument("gate_id")
    for name in ("repair-status", "repair-recover"):
        sub.add_parser(name).add_argument("run_id")
    args = parser.parse_args(argv)
    if args.command.startswith("repair-"):
        from driversi.rsi.correction.store import RepairStore
        store = RepairStore(args.registry)
        if args.command == "repair-gate":
            from driversi.rsi.correction.gate import run_gate
            result = run_gate(store, read_mapping(args.file), args.output, base=args.file.parent)
        elif args.command == "repair-run":
            from driversi.rsi.correction.recipes import run
            result = run(store, read_mapping(args.file), args.output, base=args.file.parent)
        elif args.command == "repair-gate-status":
            result = store.get_gate(args.gate_id)
        elif args.command == "repair-gate-recover":
            result = store.recover_gate(args.gate_id)
        elif args.command == "repair-recover":
            result = store.recover(args.run_id)
        else:
            result = store.get(args.run_id)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["status"] == "completed" else 2
    if args.command == "admit-teacher-search":
        from driversi.rsi.contracts import _keys
        from driversi.rsi.correction.dataset import build_search_dataset

        request = read_mapping(args.file)
        _keys(request, ("search_spec", "attempts", "policy", "navtrain_membership", "protected_groups"),
              "teacher search publication")
        ledger = ExperimentRegistry(args.registry, read_only=True)
        protected = set(request["protected_groups"]) | set(ledger.protected_groups())
        result = build_search_dataset(request["search_spec"], request["attempts"], request["policy"],
                                      request["navtrain_membership"], protected, args.output)
        print(json.dumps(result["summary"], ensure_ascii=False, allow_nan=False))
        return 0
    if args.command.startswith("campaign-"):
        from driversi.rsi.experiments.campaign import recover_campaign, register_campaign, run_campaign
        from driversi.rsi.experiments.campaign_store import CampaignStore
        if args.command == "campaign-register":
            result = register_campaign(args.registry, args.file, args.output_root)
        elif args.command == "campaign-run":
            result = run_campaign(args.registry, args.campaign_id)
        elif args.command == "campaign-recover":
            result = recover_campaign(args.registry, args.campaign_id)
        else:
            store = CampaignStore(args.registry)
            if args.command == "campaign-cancel":
                store.cancel(args.campaign_id)
            result = store.get(args.campaign_id)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 2 if args.command == "campaign-run" and result["status"] != "completed" else 0
    if args.command == "execute":
        from driversi.rsi.execution.executor import execute
        result = execute(args.registry, args.file, args.output_root)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["status"] == "completed" else 2
    if args.command in ("status", "cancel", "recover"):
        from driversi.rsi.execution.jobs import JobStore
        store = JobStore(args.registry)
        if args.command == "cancel":
            store.cancel(args.job_id)
        elif args.command == "recover":
            from driversi.rsi.execution.executor import recover
            recover(store, args.job_id)
        print(json.dumps(store.get(args.job_id), ensure_ascii=False, allow_nan=False))
        return 0
    registry = ExperimentRegistry(args.registry, read_only=args.command == "list")
    if args.command == "register":
        print(registry.register(read_mapping(args.file)))
    elif args.command == "record":
        print(registry.record(read_mapping(args.file)))
    elif args.command == "list":
        print(json.dumps(registry.public_summary(), ensure_ascii=False, indent=2))
    else:
        print("Empty experiment ledger ready; no jobs launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
