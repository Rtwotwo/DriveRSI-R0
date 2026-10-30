"""Small real-world native inference check; not a DriveRL learning test."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from driversi.agents.backends.sources import require_source

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "agents/driverl/native/DriveRL/src"),
    str(ROOT / "agents/driverl/native/DriveRL/nuplan-devkit"),
]


def main():
    from driversi.agents.backends.native import digest
    from driversi.contracts.scene import Scene
    from driversi_agents.driverl.closed_loop import DriveRLSession

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        raise ValueError("Pilot supports 1-100 environment steps")
    if not args.output.resolve().is_relative_to(root / "outputs"):
        raise ValueError("Output must be under project outputs")
    cfg = yaml.safe_load((root / "src/config/agents/driverl/closed_loop.yaml").read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    provenance = {
        "model_config_sha256": digest(root / "src/config/agents/driverl/closed_loop.yaml"),
        "checkpoint": str(root / cfg["checkpoint"]),
        "checkpoint_sha256": digest(root / cfg["checkpoint"]),
        "upstream_revision": require_source(root / "agents/driverl/native")["actual"],
        "device": args.device,
        "seed": 42,
        "requested_steps_per_log": args.steps,
    }
    rows = []
    for name in ("log-03", "log-05"):
        src = (
            root / "outputs/demonstrations/driverl-multicity-v1" / name / "prepared-final/scene.npz"
        )
        started = time.monotonic()
        session = None
        row = {"log": name, "scene": str(src), "learner_available": False, "optimizer_steps": 0}
        try:
            scene = Scene.load(src)
            session = DriveRLSession(
                scene,
                repository=root / cfg["repository"],
                checkpoint=root / cfg["checkpoint"],
                maps_root=cfg["maps_root"],
                device=args.device,
                parity=True,
            )
            actions = []
            for tick in range(args.steps):
                if session.world.episode_done:
                    break
                session.step()
                actions.append(
                    {"tick": tick, "parity": session.parity[-1], "decision": session.last_decision}
                )
            row.update(
                status="completed",
                environment_steps=len(actions),
                native_output_max_abs_error=max(
                    x["parity"].get("native_output_max_abs_error", 0) for x in actions
                ),
                contacts=session.world.contacts(),
                source_input_parity=session.source_parity,
            )
            (out / (name + ".json")).write_text(
                json.dumps(
                    actions,
                    indent=2,
                    default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
                )
            )
        except Exception as e:
            row.update(status="failed", error=repr(e))
        finally:
            if session:
                session.close()
        row["wall_seconds"] = time.monotonic() - started
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
    provenance["runtime_imports"] = {
        name: str(Path(module.__file__).resolve())
        for name, module in list(sys.modules.items())
        if getattr(module, "__file__", None) and name.split(".")[0] in ("driverl", "nuplan")
    }
    if any("/clones/" in path for path in provenance["runtime_imports"].values()):
        raise RuntimeError("Model runtime imported reference clone code")
    (out / "report.json").write_text(
        json.dumps(
            {
                "results": rows,
                "scope": "inference_and_closed_loop_only",
                "provenance": provenance,
                "training_verified": False,
                "model_performance_verified": False,
            },
            indent=2,
            default=str,
        )
    )

    return 0 if all(row["status"] == "completed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
