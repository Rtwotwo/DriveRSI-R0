"""Geometric contact evidence, separated from selected-agent driving scores."""

from collections import Counter

import numpy as np


def contact_report(scene, states, presence, lifecycle, owner):
    """Classify first observed pair contact, without dropping or assigning blame.

    Contact at an anchor may precede the recorded suffix. Entry overlap means a
    newly present object overlaps the current world, not proof of a source error.
    """
    if owner not in scene.actor_ids:
        raise ValueError("Contact owner must be a world actor")
    if not (len(states) == len(presence) == len(lifecycle)):
        raise ValueError("Contact evidence must cover every recorded state")
    indices = {a: i for i, a in enumerate(scene.actor_ids)}
    roles = scene.metadata.get("vehicle_fleet", {}).get("roles", {})
    pairs, owner_ticks, other_ticks = {}, 0, 0
    for k, frame in enumerate(lifecycle):
        contacts = frame.get("contacts", [])
        owner_ticks += any(owner in pair for pair in contacts)
        other_ticks += any(owner not in pair for pair in contacts)
        for raw in contacts:
            pair = tuple(sorted(raw))
            if len(pair) != 2 or pair[0] == pair[1] or any(a not in indices for a in pair):
                raise ValueError("Contact pair must reference two distinct world actors")
            if pair not in pairs:
                first_tick = int(frame["tick"])
                ii = [indices[a] for a in pair]
                category = (
                    "initial_overlap"
                    if k == 0 and first_tick == 0
                    else "existing_at_anchor"
                    if k == 0
                    else "entry_overlap"
                    if any(not presence[k - 1][i] for i in ii)
                    else "execution_contact"
                )
                pairs[pair] = {
                    "actors": list(pair),
                    "first_tick": first_tick,
                    "last_tick": first_tick,
                    "contact_frames": 0,
                    "onset": category,
                    "involves_owner": owner in pair,
                    "kinds": [int(scene.kinds[i]) for i in ii],
                    "control_modes": [roles.get(a, {}).get("mode", "other") for a in pair],
                    "first_speeds_mps": [float(np.linalg.norm(states[k][i, 4:6])) for i in ii],
                }
            pairs[pair]["contact_frames"] += 1
            pairs[pair]["last_tick"] = int(frame["tick"])
    return {
        "version": 1,
        "owner": owner,
        "semantics": "geometric_evidence_not_PDM_or_fault",
        "owner_contact_ticks": int(owner_ticks),
        "other_contact_ticks": int(other_ticks),
        "onset_counts": dict(Counter(p["onset"] for p in pairs.values())),
        "pairs": [pairs[k] for k in sorted(pairs)],
    }
