"""Additional, versioned per-branch EPDMS admission; never aliases v1 PDMS."""
import json
import math
import os
from pathlib import Path
import subprocess

from driversi.agents.backends.native import digest, project_root

PROTOCOL = "navsim-v2-native-complete-epdms-executed-counterfactual-v1"
POLICY = dict(metric="EPDMS", scope="each_generated_trajectory", score_unit="0_to_1",
              operator=">", threshold=0.95, protocol=PROTOCOL,
              student_benchmark_mean=False, missing_score="reject", extended_comfort="required", weighted_sum_denominator=16.0)
MULTIPLIERS = ("no_at_fault_collisions", "drivable_area_compliance",
               "driving_direction_compliance", "traffic_light_compliance")
WEIGHTS = dict(ego_progress=5.0, time_to_collision_within_bound=5.0,
               lane_keeping=2.0, history_comfort=2.0, two_frame_extended_comfort=2.0)


def scorer_sources():
    root = project_root()
    files = list((root / "src/evaluation/epdms").glob("*.py"))
    files += list((root / "src/vendor/driving/navsim-v2.2/navsim/planning/simulation/planner/pdm_planner").rglob("*.py"))
    files += [root / "src/rsi/correction/epdms_admission.py"]
    return {str(p.resolve()): digest(p) for p in files}


def validate_score(value, *, require_pass=True):
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL:
        raise ValueError("Missing native counterfactual EPDMS evidence")
    if (value.get("kind") != "executed_counterfactual_epdms"
            or value.get("score_unit") != "0_to_1"
            or value.get("native_metric_math_unmodified") is not True
            or value.get("logged_benchmark") is not False
            or value.get("actual_world_seconds") != 5.0
            or value.get("scored_seconds") != 4.0
            or value.get("history_origin") != "actual_generated_world"
            or value.get("observation_origin") != "actual_executed_counterfactual_world"
            or value.get("original_human_filter_applied") is not False
            or value.get("two_frame_comfort_policy") != "required_native_adjacent_plans"):
        raise ValueError("EPDMS input or aggregation contract differs")
    parts = value.get("components", {})
    numbers = [value.get("score")] + [parts.get(k) for k in (*MULTIPLIERS, *WEIGHTS)]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in numbers):
        raise ValueError("EPDMS requires all finite native components in [0,1]")
    comfort = value.get("two_frame_extended_comfort")
    if (type(comfort) not in (int, float) or comfort not in (0.0, 1.0)
            or comfort != parts["two_frame_extended_comfort"]):
        raise ValueError("Native extended comfort must be present and consistent")
    expected = math.prod(parts[k] for k in MULTIPLIERS) * sum(parts[k] * w for k, w in WEIGHTS.items()) / sum(WEIGHTS.values())
    if not math.isclose(expected, value["score"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("EPDMS does not match full native aggregation including EC")
    if value.get("strict_threshold") != 0.95 or value.get("passed") is not (value["score"] > 0.95):
        raise ValueError("EPDMS strict threshold receipt differs")
    if require_pass and value["score"] <= 0.95:
        raise ValueError("Each synthetic trajectory requires EPDMS strictly greater than 95")
    return value


def required_for(output):
    path = project_root() / "dataset/collection.json"
    if not path.exists():
        return False
    config = json.loads(path.read_text())
    policy = config.get("trajectory_quality")
    if policy is None:
        return False
    if policy != POLICY:
        raise ValueError("Unsupported active trajectory quality policy")
    return Path(output).resolve().is_relative_to(Path(config["campaign"]).resolve())


def run(context_path, output):
    """Separate owned v2 process: DriveRL's loaded v1 modules remain untouched."""
    root = project_root()
    python = root / ".venvs/augustus/bin/python"
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONNOUSERSITE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    command = [str(python), "-m", "driversi.evaluation.epdms.executed", "--context",
               str(context_path), "--context-sha256", digest(context_path), "--output", str(output)]
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError("Native EPDMS worker failed: " + result.stderr[-3000:])
    return validate_score(json.loads(Path(output).read_text()), require_pass=False)


def validate_sample(sample):
    """Recheck the same physical branch and scorer files sealed by native admission."""
    if sample.get("kind") != "chainflow_admitted_synthetic_sample":
        raise ValueError("Current EPDMS collection is bound to ChainFlow native supervision")
    native = sample["row"]["admission"]
    if digest(native["path"]) != native["sha256"]:
        raise ValueError("Native branch admission changed")
    proof = json.loads(Path(native["path"]).read_text())
    branch = proof["branch"]
    report = json.loads(Path(branch["report"]).read_text())
    value = validate_score(report["trajectory_evidence"]["score"].get("epdms"))
    expected = {str(Path(branch[k]).resolve()): proof["files"][branch[k]] for k in ("scene", "rollout")}
    # The renderer seals byte-identical copies in its task directory. Paths may
    # change, but roles and contents must not. Keep checking the scored originals
    # as well as the copies; a relocation is not a new physical branch.
    originals = value.get("branch_files")
    if not isinstance(originals, dict) or len(originals) != 2:
        raise ValueError("EPDMS belongs to a different physical branch")
    names = [Path(path).name for path in expected]
    if len(set(names)) != 2:
        raise ValueError("EPDMS branch roles must be distinct")
    for path, sha in expected.items():
        matches = [(source, value) for source, value in originals.items()
                   if Path(source).name == Path(path).name]
        if len(matches) != 1 or matches[0][1] != sha:
            raise ValueError("EPDMS belongs to a different physical branch")
    context = value["context"]
    capture = value["planning_capture"]
    files = {**value["source_files"], **originals, **expected, context["path"]: context["sha256"],
             capture["path"]: capture["sha256"]}
    if value["source_files"] != scorer_sources():
        raise ValueError("EPDMS must bind the complete owned evaluation source")
    if any(digest(p) != sha for p, sha in files.items()):
        raise ValueError("EPDMS branch, context or scoring code changed")
    return dict(protocol=PROTOCOL, score=value["score"], score_unit="0_to_1", threshold=0.95, operator=">")
