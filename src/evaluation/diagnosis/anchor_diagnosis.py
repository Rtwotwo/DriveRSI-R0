"""Same-anchor controlled comparisons; evidence categories are not accident blame."""

import numpy as np

from driversi.contracts.map_support import crop_support
from driversi.evaluation.metrics.contacts import contact_report
from driversi.evaluation.metrics.geometry import rectangle_corners, rectangles_overlap
from driversi.evaluation.metrics.source_support import source_entry_overlaps
from driversi.sim.execution.continuous import ContinuousSession
from driversi.sim.persistence.recording import RolloutRecord


def geometry_contacts(scene, state, present, owner):
    i = scene.actor_ids.index(owner)
    if not present[i]:
        return set()
    first = rectangle_corners(*state[i, :2], state[i, 3], *scene.dimensions[i, :2])
    result = set()
    for j in np.flatnonzero(present):
        if (
            i == j
            or np.linalg.norm(state[i, :2] - state[j, :2])
            > 0.5 * np.hypot(*scene.dimensions[[i, j], :2].T).sum()
        ):
            continue
        second = rectangle_corners(*state[j, :2], state[j, 3], *scene.dimensions[j, :2])
        if rectangles_overlap(first, second):
            result.add(tuple(sorted((owner, scene.actor_ids[j]))))
    return result


def record_evidence(scene, record, owner):
    contacts = contact_report(scene, record.states, record.presence, record.lifecycle, owner)
    disagreement = []
    map_outside = []
    source_contacts = []
    map_warnings = []
    for k, (state, present, life) in enumerate(
        zip(record.states, record.presence, record.lifecycle)
    ):
        native = {tuple(sorted(p)) for p in life["contacts"] if owner in p}
        independent = geometry_contacts(scene, state, present, owner)
        if native != independent:
            disagreement.append(
                {"tick": life["tick"], "native": sorted(native), "SAT": sorted(independent)}
            )
        support_report = crop_support(scene, state, present)
        if support_report["actors"]:
            map_warnings.append({"tick": life["tick"], **support_report})
            if any(a["actor"] == owner for a in support_report["actors"]):
                map_outside.append(life["tick"])
        tick = life["tick"]
        source = np.zeros_like(state)
        source[:, :2], source[:, 3] = scene.poses[tick, :, :2], scene.poses[tick, :, 3]
        support = (
            np.asarray(
                scene.metadata.get("reactive_background", {}).get("source_presence", scene.presence)
            )[tick]
            if scene.presence is not None
            else present
        )
        pairs = geometry_contacts(scene, source, support, owner)
        if pairs:
            source_contacts.append({"tick": tick, "pairs": sorted(pairs)})
    onset_evidence = []
    raw_support = np.asarray(
        scene.metadata.get("reactive_background", {}).get(
            "source_presence", [scene.presence_at(t) for t in range(len(scene.times))]
        )
    )
    for pair in contacts["pairs"]:
        tick = pair["first_tick"]
        ids = [scene.actor_ids.index(a) for a in pair["actors"]]
        supported = bool(raw_support[tick, ids].all())
        source_overlap = None
        if supported:
            boxes = [
                rectangle_corners(
                    *scene.poses[tick, i, :2], scene.poses[tick, i, 3], *scene.dimensions[i, :2]
                )
                for i in ids
            ]
            source_overlap = bool(rectangles_overlap(*boxes))
        onset_evidence.append(
            {
                "actors": pair["actors"],
                "tick": tick,
                "onset": pair["onset"],
                "involves_owner": pair["involves_owner"],
                "source_both_annotated": supported,
                "source_reference_overlap": source_overlap,
                "category": "source_reference_overlap"
                if source_overlap
                else "source_annotation_support_missing"
                if not supported
                else "simulated_contact_not_in_source_reference",
                "causal_blame": "not_established",
            }
        )
    return {
        "contacts": contacts,
        "contact_onset_evidence": onset_evidence,
        "geometry_disagreements": disagreement,
        "near_or_outside_map_crop_ticks": map_outside,
        "all_actor_map_support_warnings": map_warnings,
        "source_reference_contacts": source_contacts,
        "source_comparison_scope": "same source time; source pose is NOT the counterfactual anchor",
    }


def diagnose_anchor(parent, checkpoint, *, owner, steps=40, brake_steps=10):
    if owner not in parent.world.actor_ids:
        raise ValueError("Intervention owner must have a policy action slot")
    if type(steps) is not int or not 1 <= steps <= 200 or not 1 <= brake_steps <= steps:
        raise ValueError("Diagnosis requires bounded1..200steps")
    before = parent.snapshot()
    records, evidence, finals, features = {}, {}, {}, {}
    for name in ("baseline", "repeat", "brake"):
        child = ContinuousSession(
            parent.world._scene,
            parent.policy,
            history_steps=parent.history.maxlen,
            window_steps=parent.configuration["window_steps"],
        )
        try:
            child.restore(checkpoint)
            record = RolloutRecord.start(child.world)
            features[name] = []
            for j in range(steps):
                if child.world.episode_done:
                    break
                transition = child.step({owner: 1} if name == "brake" and j < brake_steps else None)
                record.append(
                    child.world,
                    child.last_decision["executed_actions"],
                    transition,
                    child.last_decision["diagnostics"],
                )
                features[name].append(transition.observation.features.copy())
            records[name] = record
            evidence[name] = record_evidence(child.world._scene, record, owner)
            finals[name] = child.snapshot()
        finally:
            child.close()
    a, b = records["baseline"], records["repeat"]
    repeat_exact = (
        np.array_equal(a.states, b.states)
        and np.array_equal(a.motion, b.motion)
        and np.array_equal(a.actions, b.actions)
        and np.array_equal(a.presence, b.presence)
        and np.array_equal(features["baseline"], features["repeat"])
        and finals["baseline"] == finals["repeat"]
        and all(
            np.array_equal(getattr(x, key), getattr(y, key))
            for x, y in zip(a.transitions, b.transitions)
            for key in ("rewards", "valid", "terminated", "truncated")
        )
    )
    baseline_count = evidence["baseline"]["contacts"]["owner_contact_ticks"]
    brake_count = evidence["brake"]["contacts"]["owner_contact_ticks"]
    flags = []
    source_integrity = source_entry_overlaps(parent.world._scene)
    if any(e["tick"] <= checkpoint["world"]["tick"] for e in source_integrity["events"]):
        flags.append("source_entry_overlap_precedes_anchor_requires_review")
    if not repeat_exact:
        flags.append("platform_execution_not_reproducible")
    if evidence["baseline"]["geometry_disagreements"]:
        flags.append("collision_geometry_disagreement_requires_inspection")
    if evidence["baseline"]["near_or_outside_map_crop_ticks"]:
        flags.append("map_support_boundary")
    if any(
        p["involves_owner"]
        and p["onset"] in {"initial_overlap", "existing_at_anchor", "entry_overlap"}
        for p in evidence["baseline"]["contacts"]["pairs"]
    ):
        flags.append("anchor_or_entry_overlap_confounder")
    action_changed = not np.array_equal(a.actions, records["brake"].actions)
    if repeat_exact and action_changed and baseline_count != brake_count:
        flags.append("contact_sensitive_to_ego_action")
    if not flags:
        flags.append("no_owner_contact_in_window" if not baseline_count else "cause_unresolved")
    if parent.snapshot() != before:
        raise RuntimeError("Diagnostic branches mutated parent session")
    return {
        "version": "same-anchor-diagnosis-v1",
        "source_integrity": source_integrity,
        "anchor_tick": checkpoint["world"]["tick"],
        "owner": owner,
        "requested_steps": steps,
        "brake_steps": brake_steps,
        "repeat_exact": bool(repeat_exact),
        "action_changed": bool(action_changed),
        "evidence_flags": flags,
        "branches": evidence,
        "blame": "not_established",
        "official_PDM": None,
        "trainable": False,
        "interpretation": "Action sensitivity does not prove model fault; source replay is a confounder check, not a reactive alternative.",
    }, records
