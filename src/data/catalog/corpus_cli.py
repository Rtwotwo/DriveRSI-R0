"""Run full feature coverage first, then an explicitly released caption profile."""

import argparse
import json
from pathlib import Path

from driversi.data.catalog.corpus import dump_corpus, write_report
from driversi.data.catalog.labeling import annotator_profile, label_batch
from driversi.data.catalog.store import Catalog, identity


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("request")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    if set(request["databases"]) != {"navtrain", "navtest"}:
        raise ValueError("Full corpus requires both separate navtrain and navtest databases")
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    state = {"status": "features_running", "stage": "features", "features": {},
             "caption_batches": {}, "training_started": False}
    state_path = output / "progress.json"
    write_report(state_path, state)
    try:
        cfg = json.loads(Path(request["feature_config"]).read_text())
        if request.get("devices"):
            from driversi.data.catalog.parallel import dump_parallel

            state["feature_progress"] = str(output / "parallel-features" / "progress.json")
            write_report(state_path, state)
            report = dump_parallel(request["databases"], cfg, output / "parallel-features",
                                   devices=request["devices"], workers=request["workers"])
            state["features"] = {split: {**value, "profile": report["profile"]}
                                 for split, value in report["splits"].items()}
            write_report(state_path, state)
            if not report["complete"]:
                raise RuntimeError("Feature coverage interrupted; resume before captions")
        else:
            for split in ("navtrain", "navtest"):
                catalog = Catalog(request["databases"][split], split)
                try:
                    report = dump_corpus(catalog, cfg, output / (split + "-features"), workers=request["workers"])
                finally:
                    catalog.close()
                state["features"][split] = report
                write_report(state_path, state)
                if not report["complete"]:
                    raise RuntimeError("Feature coverage interrupted; resume before captions")
        state.update(status="features_accounted", stage="caption_gate")
        write_report(state_path, state)
        release_path = Path(request["caption_release"])
        if not release_path.is_file():
            state["status"] = "features_done_caption_quality_review_required"
            write_report(state_path, state)
            return
        config = json.loads(Path(request["caption_config"]).read_text())
        release = json.loads(release_path.read_text())
        profile = identity(annotator_profile(config))
        if release.get("profile") != profile or release.get("batch_observation_annotation_allowed") is not True:
            raise ValueError("Caption quality release does not match current profile")
        state.update(status="captions_running", stage="captions", caption_profile=profile)
        for split in ("navtrain", "navtest"):
            catalog = Catalog(request["databases"][split], split)
            try:
                feature_profile = state["features"][split]["profile"]
                tokens = [r[0] for r in catalog.db.execute(
                    "SELECT token FROM vectors WHERE profile=? AND status='ready' ORDER BY token", (feature_profile,))]
                excluded = [dict(r) for r in catalog.db.execute(
                    "SELECT token,error FROM vectors WHERE profile=? AND status!='ready' ORDER BY token", (feature_profile,))]
                write_report(output / (split + "-caption-exclusions.json"), {"excluded": excluded})
                state["caption_batches"][split] = []
                for i in range(0, len(tokens), 64):
                    target = output / (split + "-captions") / f"batch-{i//64:04d}"
                    state["current_caption"] = {"split": split, "batch": i//64, "output": str(target)}
                    write_report(state_path, state)
                    report = label_batch(catalog, config, tokens[i:i+64], target, max_scenes=64, max_seconds=3600)
                    state["caption_batches"][split].append({"batch": i//64, "complete": report["complete"],
                        "requested": report["requested"], "validated_or_reused": sum(r["status"] in ("validated", "reused") for r in report["rows"]),
                        "failed": sum(r["status"] == "failed" for r in report["rows"])})
                    write_report(state_path, state)
                    if not report["complete"]:
                        raise RuntimeError("Caption batch interrupted; rerun to resume")
            finally:
                catalog.close()
        state.update(status="complete_with_explicit_exclusions", stage="complete")
        write_report(state_path, state)
    except Exception as exc:
        state.update(status="stopped", error=type(exc).__name__ + ": " + str(exc))
        write_report(state_path, state)
        raise


if __name__ == "__main__":
    main()
