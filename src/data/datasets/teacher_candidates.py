"""Admission of teacher supervision; this module neither generates nor scores trajectories.

DriveTTO and privileged-teacher producers share this protocol. A trusted scorer writes
versioned evidence; an immutable overlay references verified artifacts without changing
source data. Qualification here is engineering evidence, not a model-quality claim.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from driversi.rsi.contracts import _identity, _keys
from driversi.training.sampling.environment import source_log_group


def _artifact(path, digest):
    if not Path(path).is_absolute():
        raise ValueError("Artifact paths must be absolute server paths")
    p = Path(path).resolve(strict=True)
    if not p.is_file():
        raise ValueError("Artifact must be a file")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != digest:
        raise ValueError("Artifact content changed")
    return p


def validate_admission_policy(policy):
    _keys(
        policy,
        (
            "id",
            "metric",
            "protocol_version",
            "scorer_version",
            "min_score",
            "min_gain",
            "min_horizon_seconds",
            "required_views",
        ),
        "teacher admission",
    )
    for key in ("id", "protocol_version", "scorer_version"):
        _identity(policy[key], key)
    if policy["metric"] not in ("pdms", "epdms"):
        raise ValueError("Declare PDMS or EPDMS; native reward cannot substitute")
    for key in ("min_score", "min_gain", "min_horizon_seconds"):
        x = policy[key]
        if type(x) not in (float, int) or not math.isfinite(x) or x < 0:
            raise ValueError("Finite nonnegative thresholds required")
    if policy["min_score"] >= 1 or policy["min_gain"] > 1 or policy["min_horizon_seconds"] <= 0:
        raise ValueError("Scores use [0,1]; horizon must be positive")
    views = policy["required_views"]
    if not isinstance(views, list) or not views or len(views) != len(set(views)):
        raise ValueError("Declare unique required camera views")
    for view in views:
        _identity(view, "view")
    return policy


def _validate_trajectory_candidate(record, policy, navtrain_membership, protected_groups, require_positive):
    """Validate executed trajectory evidence; this alone does not admit visual supervision."""
    validate_admission_policy(policy)
    _keys(
        record,
        (
            "schema_version",
            "id",
            "source_token",
            "source_log",
            "source_split",
            "episode_id",
            "branch_id",
            "observation_state_id",
            "observation_tick",
            "state_origin",
            "teacher",
            "trajectory",
            "score",
            "execution",
            "images",
        ),
        "teacher candidate",
    )
    if type(record["schema_version"]) is not int or record["schema_version"] != 2:
        raise ValueError("Unknown teacher candidate schema")
    for key in ("id", "source_token", "episode_id", "branch_id", "observation_state_id"):
        _identity(record[key], key)
    if type(record["observation_tick"]) is not int or record["observation_tick"] < 0:
        raise ValueError("Invalid observation tick")
    if record["state_origin"] not in ("logged", "counterfactual"):
        raise ValueError("Declare observation origin")
    token = record["source_token"]
    group = source_log_group(record["source_log"])
    protected = {source_log_group(g) for g in protected_groups}
    if (
        record["source_split"] != "navtrain"
        or token not in navtrain_membership
        or source_log_group(navtrain_membership[token]) != group
        or group in protected
    ):
        raise ValueError("Candidate is not from the allowed navtrain token/log membership")
    teacher = record["teacher"]
    _keys(teacher, ("backend", "version", "training_groups", "information_set"), "teacher")
    if teacher["backend"] not in ("drivetto", "privileged_teacher"):
        raise ValueError("Unknown teacher backend")
    _identity(teacher["version"], "teacher version")
    if teacher["information_set"] != "current_state_history":
        raise ValueError("Oracle future information is diagnostic, not student supervision")
    if not isinstance(teacher["training_groups"], list) or not teacher["training_groups"]:
        raise ValueError("Teacher training lineage must be declared")
    if {source_log_group(g) for g in teacher["training_groups"]} & protected:
        raise ValueError("Teacher training overlaps protected evaluation logs")
    trajectory = record["trajectory"]
    _keys(trajectory, ("path", "sha256", "coordinate_frame", "kind"), "trajectory")
    if trajectory["coordinate_frame"] != "ego_at_observation" or trajectory["kind"] != "executed":
        raise ValueError("Require the executed ego-relative trajectory, not raw proposal")
    path = _artifact(trajectory["path"], trajectory["sha256"])
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"times", "poses"}:
            raise ValueError("Trajectory requires times and XY-heading poses")
        times, poses = data["times"], data["poses"]
        if (
            times.ndim != 1
            or len(times) < 2
            or poses.shape != (len(times), 3)
            or not np.isfinite(times).all()
            or not np.isfinite(poses).all()
            or times[0] <= 0
            or np.any(np.diff(times) <= 0)
            or times[-1] < policy["min_horizon_seconds"]
        ):
            raise ValueError("Invalid or insufficient executed trajectory horizon")
    score = record["score"]
    _keys(
        score,
        (
            "metric",
            "value",
            "reference_value",
            "scale",
            "protocol_version",
            "scorer_version",
            "evidence_path",
            "evidence_sha256",
            "context_state_id",
            "trajectory_sha256",
            "verification",
        ),
        "scorer evidence",
    )
    if (
        any(score[k] != policy[k] for k in ("metric", "protocol_version", "scorer_version"))
        or score["scale"] != "unit_interval"
        or score["context_state_id"] != record["observation_state_id"]
        or score["trajectory_sha256"] != trajectory["sha256"]
    ):
        raise ValueError("Score protocol/trajectory/state identity mismatch")
    for key in ("value", "reference_value"):
        if (
            type(score[key]) not in (int, float)
            or not math.isfinite(score[key])
            or not 0 <= score[key] <= 1
        ):
            raise ValueError("PDMS/EPDMS must use finite [0,1] scores")
    verification = score["verification"]
    _keys(verification, ("role", "mode", "world_context", "reference_trajectory", "baseline_trajectory"),
          "final verification")
    if (verification["role"] != "final_verifier"
            or verification["mode"] != "independent_fixed_reference_pair"):
        raise ValueError("Search-pool scores cannot substitute for independent fixed-reference verification")
    # The two pair evaluations use the same world/cache and fixed reference;
    # baseline trajectory is not the cached PDM reference itself.
    for name in ("world_context", "reference_trajectory", "baseline_trajectory"):
        entry = verification[name]
        _keys(entry, ("path", "sha256"), name)
        _artifact(entry["path"], entry["sha256"])
    context = json.loads(Path(verification["world_context"]["path"]).read_text())
    if any(context.get(k) != record[k] for k in
           ("source_token", "episode_id", "branch_id", "observation_state_id", "observation_tick", "state_origin")):
        raise ValueError("Scoring world context differs from the candidate observation")
    evidence = json.loads(_artifact(score["evidence_path"], score["evidence_sha256"]).read_text())
    expected = {
        k: score[k]
        for k in (
            "metric",
            "value",
            "reference_value",
            "scale",
            "protocol_version",
            "scorer_version",
            "context_state_id",
            "trajectory_sha256",
            "verification",
        )
    }
    expected.update(candidate_id=record["id"], source_token=token, branch_id=record["branch_id"])
    if evidence.get("valid") is not True or any(evidence.get(k) != v for k, v in expected.items()):
        raise ValueError("Scorer artifact does not match submitted evidence")
    if require_positive and (
        score["value"] <= policy["min_score"]
        or score["value"] - score["reference_value"] < policy["min_gain"]
    ):
        raise ValueError("Teacher score/gain does not pass the frozen threshold")
    execution = record["execution"]
    _keys(
        execution,
        ("collision_free", "road_supported", "finite_dynamics", "trajectory_executed"),
        "execution",
    )
    if any(type(v) is not bool for v in execution.values()):
        raise ValueError("Execution checks must be boolean")
    required = execution.values() if require_positive else (
        execution["road_supported"], execution["finite_dynamics"], execution["trajectory_executed"])
    if any(v is not True for v in required):
        raise ValueError("Executed-trajectory checks failed")
    return record


def _validate_images(record, policy):
    images = record["images"]
    if not isinstance(images, list) or not images:
        raise ValueError("Image supervision cannot be empty")
    views = set()
    for image in images:
        _keys(
            image,
            (
                "path",
                "sha256",
                "view",
                "observation_state_id",
                "episode_id",
                "branch_id",
                "observation_tick",
                "source",
                "asset_version",
                "alignment_valid",
                "support_valid",
            ),
            "image",
        )
        if image["view"] in views:
            raise ValueError("Duplicate camera view")
        views.add(image["view"])
        if any(
            image[k] != record[k]
            for k in ("observation_state_id", "episode_id", "branch_id", "observation_tick")
        ):
            raise ValueError("Images must be inputs at this observation, not future label frames")
        if image["alignment_valid"] is not True or image["support_valid"] is not True:
            raise ValueError("Image alignment/support not qualified")
        if image["source"] == "original_log":
            if record["state_origin"] != "logged" or image["asset_version"] is not None:
                raise ValueError("Original image cannot label a counterfactual observation")
        elif image["source"] == "rendered":
            _identity(image["asset_version"], "render asset version")
        else:
            raise ValueError("Unknown image origin")
        from PIL import Image

        with Image.open(_artifact(image["path"], image["sha256"])) as pixels:
            if min(pixels.size) <= 0:
                raise ValueError("Empty camera image")
            pixels.verify()
    if not set(policy["required_views"]).issubset(views):
        raise ValueError("Required camera view is missing")
    return record


def validate_trajectory_candidate(record, policy, navtrain_membership, protected_groups):
    return _validate_trajectory_candidate(record, policy, navtrain_membership, protected_groups, True)


def validate_comparison_candidate(record, policy, navtrain_membership, protected_groups):
    """Valid observation and score evidence, even for a collision or low-scoring trajectory."""
    _validate_trajectory_candidate(record, policy, navtrain_membership, protected_groups, False)
    _validate_images(record, policy)
    return record


def validate_candidate(record, policy, navtrain_membership, protected_groups):
    """Admit only complete trajectory AND observation evidence for supervision."""
    validate_trajectory_candidate(record, policy, navtrain_membership, protected_groups)
    _validate_images(record, policy)
    return record


def _teacher_overlay(records, policy, navtrain_membership, protected_groups):

    """Choose best qualified candidate per observation; never overwrite original data.

    All candidates (including failures) are recorded. Paths are content-bound and must
    remain available; downstream loaders should revalidate before training.
    """
    validate_admission_policy(policy)
    accepted, rejected, seen = {}, [], set()
    verification_contexts = {}
    for record in records:
        cid = record.get("id")
        if not isinstance(cid, str) or cid in seen:
            raise ValueError("Candidates require unique identities")
        seen.add(cid)
        try:
            validate_candidate(record, policy, navtrain_membership, protected_groups)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            rejected.append({"id": cid, "reason": str(exc), "candidate": record})
            continue
        key = (record["source_token"], record["observation_state_id"], record["branch_id"])
        verification = (record["score"]["verification"], record["score"]["reference_value"])
        if key in verification_contexts and verification_contexts[key] != verification:
            raise ValueError("Candidates for one observation have different fixed scoring contexts")
        verification_contexts[key] = verification
        prior = accepted.get(key)
        if prior and prior["score"]["value"] >= record["score"]["value"]:
            rejected.append(
                {"id": cid, "reason": "lower_score_same_observation", "candidate": record}
            )
        else:
            if prior:
                rejected.append(
                    {"id": prior["id"], "reason": "superseded_same_observation", "candidate": prior}
                )
            accepted[key] = record
    manifest = {
        "schema_version": 1,
        "kind": "navtrain_teacher_overlay",
        "policy": policy,
        "source_data_modified": False,
        "generated_by_this_module": False,
        "admitted": [
            {
                "replacement_key": {
                    "source_token": k[0],
                    "observation_state_id": k[1],
                    "branch_id": k[2],
                },
                "candidate": r,
            }
            for k, r in accepted.items()
        ],
        "rejected": rejected,
        "attempted": len(seen),
        "accepted": len(accepted),
        "approved_teacher": False,
    }
    return manifest


def _publish_new(path, value):
    """Publish a complete immutable manifest; readers never observe partial JSON."""
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".teacher-", suffix=".tmp", delete=False) as f:
            name = f.name
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(name, path)
        except FileExistsError as exc:
            raise ValueError("Dataset version already exists") from exc
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def build_teacher_overlay(records, policy, navtrain_membership, protected_groups, output):
    manifest = _teacher_overlay(records, policy, navtrain_membership, protected_groups)
    _publish_new(output, manifest)
    return manifest
