#!/usr/bin/env python3
"""Full CPU-only cache acceptance for one of the three Stage2 splits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["navtrain", "navtrainval", "navtrain_simscale"])
    parser.add_argument("--cache-base", default=None)
    parser.add_argument("--real-root", default=None)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(repo))
    os.environ.setdefault("NAVSIM_EXP_ROOT", str(repo / "exp"))
    os.environ.setdefault("OPENSCENE_DATA_ROOT", "/data/xinyao/navsim_workspace/dataset")
    os.environ.setdefault("NUPLAN_MAPS_ROOT", os.environ["OPENSCENE_DATA_ROOT"] + "/maps")
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    base = Path(args.cache_base or os.environ["NAVSIM_EXP_ROOT"] + "/vlm_feature_cache")
    real = Path(args.real_root or base / "navtrain_vlm_feature_cache")
    from navsim.planning.training.vlm_stage2_compatibility import source_cache_roots, verify_configured_split
    # Fail on missing physical roots before parsing the large split YAMLs.
    source_cache_roots(args.split, base, real)
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(repo / "navsim/planning/script/config/training"), version_base=None):
        cfg = compose(config_name="cache_internvl_simscale", overrides=[f"train_test_split={args.split}"])
    report = verify_configured_split(cfg, args.split, base, real)
    print(json.dumps({"split": args.split, **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
