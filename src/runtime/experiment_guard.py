"""Bounded ownership guard for benchmark feature services that outlive proxies."""

import argparse
import json
import os
import signal
import time
from pathlib import Path


def birth(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def owned_features(root):
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode().split("\0")
            if "--config" in args and any(
                x.endswith("/src/data/catalog/feature_worker.py") for x in args
            ):
                config = Path(args[args.index("--config") + 1]).resolve()
                if not config.is_relative_to(root) or not config.name.endswith(
                    ".feature-worker.json"
                ):
                    continue
                value = json.loads(config.read_text())
                if not Path(value["directory"]).resolve().is_relative_to(root):
                    continue
            elif "--job" in args and any(
                x.endswith("/src/rsi/experiments/reactive_postprocess.py") for x in args
            ):
                config = Path(args[args.index("--job") + 1]).resolve()
                if not config.is_relative_to(root):
                    continue
            elif "--output" in args and any(
                x in args
                for x in (
                    "driversi.rendering.gaussian.worldengine_replay",
                    "driversi.rendering.gaussian.worldengine_source",
                )
            ):
                config = Path(args[args.index("--output") + 1]).resolve()
                if not config.is_relative_to(root):
                    continue
            else:
                continue
            identity = birth(int(proc.name))
            if identity is not None:
                matches.append((int(proc.name), identity, str(config)))
        except (FileNotFoundError, PermissionError, IndexError, ValueError, KeyError):
            continue
    return matches


def settle(root):
    rows = []
    for pid, identity, config in owned_features(root):
        if birth(pid) != identity:
            continue
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while birth(pid) == identity and time.monotonic() < deadline:
            time.sleep(0.1)
        forced = birth(pid) == identity
        if forced:
            os.kill(pid, signal.SIGKILL)
            deadline = time.monotonic() + 10
            while birth(pid) == identity and time.monotonic() < deadline:
                time.sleep(0.1)
        rows.append(dict(pid=pid, config=config, forced=forced, settled=birth(pid) != identity))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", type=int, required=True)
    parser.add_argument("--birth", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=14400)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    project = Path(__file__).resolve().parents[2]
    if (
        root == project / "exp"
        or not root.is_relative_to(project / "exp")
        or not 1 <= args.seconds <= 14400
    ):
        raise ValueError("Explicit benchmark root and bounded guard required")
    deadline = time.monotonic() + args.seconds
    while birth(args.owner) == args.birth and time.monotonic() < deadline:
        time.sleep(1)
    if birth(args.owner) == args.birth:
        # Guard deadline never authorizes killing a still-owned active service.
        result = dict(status="guard_expired_owner_alive", rows=[])
    else:
        rows = settle(root)
        result = dict(status="owner_exited_services_settled", rows=rows)
    (root / "owner-guard-result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
