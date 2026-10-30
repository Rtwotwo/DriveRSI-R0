"""History-conditioned procedural memory for offline VLA model services.

Cases are diagnostic experience, not policy weights or causal explanations.
Only reviewed code is executable. Frozen mode never writes persistent memory.
"""

import json
import math
import sqlite3
from pathlib import Path

from driversi.data.catalog.store import identity

LIBRARY_VERSION = 1
SKILL_TREE = {
    "task": "augustus/hard_scene_service",
    "version": LIBRARY_VERSION,
    "atomics": [
        {
            "id": "world_validity",
            "base_skills": ["history_signature", "contact_trace_audit", "native_readiness_audit"],
        },
        {"id": "teacher_qualification", "base_skills": ["native_readiness_audit"]},
        {"id": "joint_supervision", "base_skills": ["native_readiness_audit"]},
    ],
    "code_consolidation": "reviewed_code_only_no_automatic_installation",
}


def membership_reference(metadata, teacher, window):
    member = metadata.get("source_train_membership", {})
    if (
        member.get("reference_role") != "official_split_membership_not_physical_anchor"
        or metadata.get("split") != "train"
        or window.get("source_split") != "train"
        or member.get("source_log") != teacher["source_log"]
        or metadata.get("source_log") != teacher["source_log"]
        or window.get("source_log") != teacher["source_log"]
        or metadata.get("scenario_id") != teacher["token"]
        or window.get("source_token") != teacher["token"]
    ):
        raise ValueError("Explicit official membership and physical start identities required")
    return member


def history_signature(window):
    """Use only actual historical geometry; never read future trajectories."""
    rows = window.get("history", [])
    if len(rows) != 4 or window.get("input_scope") != "history_only":
        raise ValueError("Four actual history frames required")
    times, poses = [], []
    for row in rows:
        t, p = row["relative_seconds"], row["ego_rear_axle_pose"]
        if len(p) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in [t, *p]):
            raise ValueError("Finite historical rear-axle poses required")
        times.append(t)
        poses.append(p)
    if any(abs(t - expected) > 1e-6 for t, expected in zip(times, [-1.5, -1.0, -0.5, 0.0])):
        raise ValueError("Native 2 Hz historical timeline required")
    speed = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poses, poses[1:])) / 1.5
    turn = sum(
        abs(math.atan2(math.sin(b[3] - a[3]), math.cos(b[3] - a[3])))
        for a, b in zip(poses, poses[1:])
    )
    objects = rows[-1]["objects"]
    if len({o["actor_id"] for o in objects}) != len(objects):
        raise ValueError("Duplicate active actor in historical anchor")
    near = 0
    for obj in objects:
        xy = obj["center_state"][:2]
        if len(xy) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in xy):
            raise ValueError("Finite historical object positions required")
        if (
            obj["actor_id"] != "ego"
            and math.hypot(xy[0] - poses[-1][0], xy[1] - poses[-1][1]) <= 30
        ):
            near += 1
    tags = [
        "ego_stopped" if speed < 0.2 else "ego_low_speed" if speed < 2 else "ego_moving",
        "history_turning" if turn > 0.1 else "history_near_straight",
        "nearby_objects_dense"
        if near >= 8
        else "nearby_objects_sparse"
        if near
        else "nearby_objects_absent",
    ]
    return {
        "conditions": sorted(tags),
        "history_mean_path_speed_mps": speed,
        "history_absolute_turn_rad": turn,
        "objects_within_30m": near,
        "information_set": "executed_history_only",
        "future_read": False,
        "semantics": "retrieval_descriptors_not_driving_failure_labels",
    }


def earliest_unpassed_gate(diagnosis):
    failed = set(diagnosis["failed_physical_checks"])
    world = {
        "arrival_contacts_absent",
        "future_contacts_absent",
        "restore_exact",
        "producer_changed_history_evidence",
    }
    if failed & world:
        return "world_validity"
    if failed:
        return "teacher_qualification"
    if diagnosis["learning_blockers"]:
        return "joint_supervision"
    return None


def contact_trace(teacher):
    """Count observed pair/tick records, not accidents or blame attribution."""
    if "arrival_contacts" not in teacher or "object_contacts" not in teacher.get(
        "branches", {}
    ).get("teacher", {}):
        raise ValueError("Complete contact traces required")
    anchor = teacher["arrival_tick"]
    if type(anchor) is not int or anchor < 0:
        raise ValueError("Actual anchor tick required")
    pairs = {}
    seen = set()
    counts = {"arrival": 0, "future": 0}
    for phase, records in [
        ("arrival", teacher["arrival_contacts"]),
        ("future", teacher["branches"]["teacher"]["object_contacts"]),
    ]:
        for record in records:
            tick, actors = record["tick"], record["actors"]
            if (
                type(tick) is not int
                or tick < 0
                or len(actors) != 2
                or len(set(actors)) != 2
                or any(not isinstance(a, str) or not a for a in actors)
            ):
                raise ValueError("Malformed contact record")
            if (phase == "arrival" and tick > anchor) or (phase == "future" and tick <= anchor):
                raise ValueError("Contact phase crosses selected anchor")
            key = tuple(sorted(actors))
            event = (tick, key)
            if event in seen:
                raise ValueError("Duplicate pair/tick contact record")
            seen.add(event)
            counts[phase] += 1
            item = pairs.setdefault(
                key,
                {
                    "actors": list(key),
                    "first_tick": tick,
                    "last_tick": tick,
                    "arrival_records": 0,
                    "future_records": 0,
                },
            )
            item["first_tick"] = min(item["first_tick"], tick)
            item["last_tick"] = max(item["last_tick"], tick)
            item[phase + "_records"] += 1
    values = sorted(pairs.values(), key=lambda x: (x["first_tick"], x["actors"]))
    return {
        "kind": "contact_trace_audit",
        "counts": counts,
        "distinct_pairs": len(values),
        "present_at_initial_tick_pairs": sum(p["first_tick"] == 0 for p in values),
        "pairs_persisting_across_anchor": sum(
            p["arrival_records"] > 0 and p["future_records"] > 0 for p in values
        ),
        "new_pairs_after_anchor": sum(p["arrival_records"] == 0 for p in values),
        "ego_pairs": sum("ego" in p["actors"] for p in values),
        "pairs": values,
        "causal_attribution": "unresolved",
        "unit": "pair_tick_records_not_independent_accidents",
    }


class SceneSkillMemory:
    def __init__(self, path, mode):
        if mode not in ("evolve", "frozen"):
            raise ValueError("Explicit evolve/frozen mode required")
        self.path, self.mode = Path(path), mode
        self.db = None
        if mode == "frozen":
            if self.path.exists():
                self.db = sqlite3.connect(
                    self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30
                )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path, timeout=30)
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS vla_scene_cases (id TEXT PRIMARY KEY,payload TEXT NOT NULL)"
            )
            self.db.commit()

    def close(self):
        if self.db:
            self.db.close()

    def cases(self):
        if self.db is None:
            return []
        return [
            json.loads(r[0])
            for r in self.db.execute("SELECT payload FROM vla_scene_cases ORDER BY id")
        ]

    def record(self, case):
        if self.mode == "frozen":
            return {"recorded": False, "reason": "frozen_no_write"}
        if case.get("student_gain_credited") is not False or case.get("source_split") != "navtrain":
            raise ValueError("Only non-credit diagnostic training-source cases allowed")
        key = case["case_id"]
        payload = json.dumps(case, sort_keys=True, allow_nan=False)
        with self.db:
            old = self.db.execute(
                "SELECT payload FROM vla_scene_cases WHERE id=?", (key,)
            ).fetchone()
            if old and old[0] != payload:
                raise ValueError("Conflicting evidence for existing memory case")
            self.db.execute("INSERT OR IGNORE INTO vla_scene_cases VALUES (?,?)", (key, payload))
        return {
            "recorded": old is None,
            "reason": "new_case" if old is None else "already_recorded",
            "case_id": key,
        }

    def retrieve(self, context, signature, source_group, limit=3):
        if type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError("Bounded memory retrieval required")
        query = set(signature["conditions"])
        matches = []
        for case in self.cases():
            if case["context"] != context or case["source_group"] == source_group:
                continue
            tags = set(case["history_signature"]["conditions"])
            shared = query & tags
            similarity = len(shared) / len(query | tags) if query | tags else 0
            if similarity < 0.25:
                continue
            matches.append(
                {
                    "case_id": case["case_id"],
                    "similarity": similarity,
                    "shared_conditions": sorted(shared),
                    "earliest_unpassed_gate": case["earliest_unpassed_gate"],
                    "source_group": case["source_group"],
                    "suggested_base_skill": "contact_trace_audit"
                    if case["contact_evidence_present"]
                    else "native_readiness_audit",
                    "evidence_scope": case["evidence_scope"],
                    "relevance_calibrated": False,
                }
            )
        matches.sort(key=lambda x: (-x["similarity"], x["case_id"]))
        return matches[:limit]


def generation_schedule(mode, matches):
    """Choose preflight work, never bypass compulsory downstream admission.

    No-memory uses the native teacher workflow without optional preflight;
    fixed always performs the contact preflight; memory performs it when a
    matching observed contact case recommends it. All use the same frozen
    DriveRL arrival/future so the comparison does not change the policy.
    """
    if mode not in ("no_memory", "fixed", "memory"):
        raise ValueError("Unknown generation comparison arm")
    if mode != "memory" and matches:
        raise ValueError("Memory-disabled arms cannot consume retrieved cases")
    applicable = [
        x
        for x in matches
        if x.get("suggested_base_skill") == "contact_trace_audit"
        and set(x.get("shared_conditions", [])) & {"nearby_objects_sparse", "nearby_objects_dense"}
    ]
    contact = mode == "fixed" or bool(applicable)
    return {
        "generation_policy_version": 2,
        "mode": mode,
        "preflight": ["initial_contact_audit"] if contact else [],
        "arrival_backend": "driverl_native",
        "teacher_backend": "driverl_native",
        "retrieved_case_ids": [x["case_id"] for x in matches],
        "applicable_case_ids": [x["case_id"] for x in applicable],
        "excluded_case_ids": [x["case_id"] for x in matches if x not in applicable],
        "applicability_rule": "contact_preflight_requires_shared_nearby_object_condition",
        "mandatory_postconditions": [
            "continuous_history",
            "pdms_ge_95",
            "physical_integrity",
            "visual_admission",
            "joint_label_review",
        ],
        "student_gain_established": False,
    }


def make_case(
    context,
    signature,
    source,
    diagnosis,
    execution_identity,
    *,
    contacts_present,
    fresh_execution=False,
    cost=None,
):
    if type(contacts_present) is not bool:
        raise ValueError("Observed contact evidence required")
    case = {
        "context": context,
        "history_signature": signature,
        **source,
        "execution_identity": execution_identity,
        "earliest_unpassed_gate": earliest_unpassed_gate(diagnosis),
        "failed_checks": diagnosis["failed_physical_checks"],
        "contact_evidence_present": contacts_present,
        "causal_attribution": "unresolved",
        "student_failure_established": False,
        "student_gain_credited": False,
        "evidence_scope": (
            "fresh_privileged_generation_not_augustus_behavior"
            if fresh_execution
            else "reused_physical_branch_model_service_not_augustus_behavior"
        ),
        "reliable_skill": False,
        "trainable": False,
    }
    if fresh_execution:
        if not isinstance(cost, dict) or set(cost) != {
            "generation_wall_seconds",
            "sim_agent_steps",
        }:
            raise ValueError("Fresh execution requires measured generation cost")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in cost.values()):
            raise ValueError("Finite nonnegative generation cost required")
        case["cost"] = cost
        case["cost_scope"] = "generation_only_excludes_render_annotation_training_evaluation"
    return {"case_id": identity(case), **case}
