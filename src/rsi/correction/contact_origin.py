"""Source geometry attribution; never a collision exemption or a scorer."""
import numpy as np
from driversi.evaluation.metrics.geometry import rectangle_corners, rectangles_overlap


def source_contacts(scene):
    ids = np.arange(len(scene.actor_ids)) if scene.presence is None else np.flatnonzero(scene.presence[0])
    boxes = {i: rectangle_corners(*scene.poses[0,i,:2], scene.poses[0,i,3],
                                  *scene.dimensions[i,:2]) for i in ids}
    kinds = scene.metadata.get("source_actor_kinds", {})
    rows = []
    for n, a in enumerate(ids):
        for b in ids[n+1:]:
            if rectangles_overlap(boxes[a], boxes[b]):
                actors = [scene.actor_ids[a], scene.actor_ids[b]]
                labels = [kinds.get(x, "unknown") for x in actors]
                rows.append(dict(actors=actors, source_kinds=labels,
                    origin="source_initial_annotation_overlap", before_policy=True,
                    static_annotation_pair=all(x in ("barrier", "traffic_cone", "czone_sign") for x in labels),
                    collision_exempt=False))
    return dict(contacts=rows, quarantined=bool(rows), trainable=False,
                reason="source_initial_overlap" if rows else None,
                attribution="source_geometry_not_model_action", scoring_modified=False)
