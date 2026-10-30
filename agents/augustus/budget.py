"""Cross-rank reserve-before-score accounting; failures retain charged attempts."""

import fcntl
import json
from pathlib import Path


def charge(path, limit, frames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        text = f.read()
        value = (
            json.loads(text)
            if text
            else {"reserved_agent_steps": 0, "scored_calls": 0, "limit": limit}
        )
        if value["limit"] != limit:
            raise ValueError("PDM budget limit changed")
        # Native v1 simulator: reference + proposal, each 40 physical steps.
        cost = frames * 80
        if value["reserved_agent_steps"] + cost > limit:
            raise RuntimeError("Harness PDM agent-step budget exhausted")
        value["reserved_agent_steps"] += cost
        value["scored_calls"] += 1
        f.seek(0)
        f.truncate()
        json.dump(value, f)
        f.flush()
        return value


def metered(function, path, limit):
    def call(preds, frame_ids, *args, **kwargs):
        if args:
            raise ValueError("Native scoring wrapper expects keyword scoring options")
        if len(preds) != len(frame_ids) or any(not x for x in frame_ids):
            raise ValueError("PDM sample identity missing")
        charge(path, limit, len(preds))
        wants_count = kwargs.get("return_count", False)
        kwargs["return_count"] = True
        result = function(preds, frame_ids, **kwargs)
        if result.get("count", 0) != len(frame_ids) or (frame_ids and "PDMS" not in result):
            raise RuntimeError("Incomplete native PDM scoring; no fallback reward allowed")
        if not wants_count:
            result.pop("count", None)
        return result

    return call
