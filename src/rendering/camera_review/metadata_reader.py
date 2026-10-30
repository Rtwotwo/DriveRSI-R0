"""Trusted MTGS pickle bridge for the isolated optional-dependency environment."""

import json
import pickle
import sys
from pathlib import Path

if __name__ == "__main__":
    request = json.load(sys.stdin)
    wanted = set(request["tokens"])
    data = pickle.loads(Path(request["path"]).read_bytes())
    output = {}
    for video in data.values():
        frames = video["frame_infos"]
        if isinstance(frames, dict):
            frames = frames.values()
        for frame in frames:
            token = str(frame["token"])
            if token not in wanted:
                continue
            if token in output:
                raise ValueError("Ambiguous reconstruction frame token")
            output[token] = dict(
                timestamp_us=int(frame["timestamp"]),
                log_name=frame["log_name"],
                refined=frame["ego2global"].tolist(),
                original=frame["ego2global_original"].tolist(),
                global_offset=video["global2world_translation"].tolist(),
                boxes=frame["gt_boxes"].tolist(),
                ids=[str(x) for x in frame["track_tokens"]],
                kinds=[str(x) for x in frame["gt_names"]],
            )
    if set(output) != wanted:
        raise ValueError("Missing reconstruction source frame")
    print(json.dumps(output, allow_nan=False))
