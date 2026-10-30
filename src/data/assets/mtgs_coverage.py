"""Exact-identity asset preflight. Coverage is necessary, never admission."""


def audit_binding_assets(binding, nodes):
    """nodes is extracted from the loaded checkpoint/runtime, not a claimed ratio.

    Every raw source annotation is retained, including stationary objects.
    No visibility exemption or cross-traversal identity reassignment is applied.
    """
    travel = binding["travel_id"]
    if type(travel) is not int or not binding.get("frames"):
        raise ValueError("Exact source binding required")
    rows = []
    for frame in binding["frames"]:
        raw_index = frame["mtgs_frame_index"]
        if (
            type(raw_index) is not int
            or raw_index < 0
            or not isinstance(frame.get("source_objects"), list)
        ):
            raise ValueError("Source frame inventory required")
        objects = frame["source_objects"]
        ids = [o["track_token"] for o in objects]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate source track identity")
        issues, matched = [], []
        for obj in objects:
            token = obj["track_token"]
            node = nodes.get(token)
            reason = None
            if node is None:
                reason = "asset_missing"
            elif node.get("travel_id") != travel:
                reason = "asset_other_traversal"
            elif type(node.get("gaussian_count")) is not int or node["gaussian_count"] <= 0:
                reason = "asset_empty_or_unverified_geometry"
            else:
                position = node.get("runtime_frame_indices", {}).get(frame["token"])
                index = None if position is None else position.get("index")
                if (
                    position is None
                    or position.get("timestamp_us") != frame["timestamp_us"]
                    or type(index) is not int
                    or index < 0
                ):
                    issues.append(
                        dict(
                            track_token=token,
                            kind=obj["kind"],
                            reason="asset_source_frame_unavailable",
                        )
                    )
                    continue
                mask = node.get("in_frame_mask")
                if not isinstance(mask, list) or any(type(v) is not bool for v in mask):
                    raise ValueError("Checkpoint node presence mask required")
                if index >= len(mask) or not mask[index]:
                    reason = "asset_absent_at_source_frame"
            if obj.get("physics_bound") is not True:
                issues.append(
                    dict(
                        track_token=token,
                        kind=obj["kind"],
                        reason="source_object_not_bound_to_world",
                    )
                )
            if reason:
                issues.append(dict(track_token=token, kind=obj["kind"], reason=reason))
            else:
                matched.append(token)
        rows.append(
            dict(
                timestamp_us=frame["timestamp_us"],
                raw_metadata_frame_index=raw_index,
                runtime_frame_indices={
                    o["track_token"]: nodes[o["track_token"]]["runtime_frame_indices"][
                        frame["token"]
                    ]["index"]
                    for o in objects
                    if o["track_token"] in matched
                },
                source_count=len(objects),
                matched_tokens=matched,
                issues=issues,
            )
        )
    return dict(
        schema_version=2,
        kind="mtgs_source_asset_preflight",
        travel_id=travel,
        source_log=binding["log_name"],
        frames=rows,
        complete=all(not r["issues"] for r in rows),
        missing_tokens=sorted({v["track_token"] for r in rows for v in r["issues"]}),
        trainable=False,
        scope="source_inventory_only_not_counterfactual_visibility_or_quality",
    )


def runtime_nodes(runtime):
    from driversi.rendering.gaussian.mtgs import _array

    # Nodes are constructed on the native train timeline, after parser filtering.
    sources = runtime.pipeline.datamanager.train_dataset._dataparser_outputs
    frames = {}
    for travel, token, index in zip(sources.travel_ids, sources.frame_tokens, sources.frame_ids):
        travel, token, index = int(travel), str(token), int(index)
        source = runtime.source_frames[(travel, token)]
        entry = dict(index=index, timestamp_us=int(source["timestamp"]))
        previous = frames.setdefault(travel, {}).setdefault(token, entry)
        if previous != entry:
            raise ValueError("Runtime cameras disagree on source frame identity")
    result = {}
    for name, node in runtime.model.gaussian_models.items():
        if name.startswith("rigid_object_"):
            result[name.removeprefix("rigid_object_")] = dict(
                travel_id=int(node.travel_id),
                gaussian_count=int(node.num_points),
                in_frame_mask=_array(node.in_frame_mask).astype(bool).tolist(),
                runtime_frame_indices=frames.get(int(node.travel_id), {}),
            )
    return result


def verify_source_inventory(binding, scene, raw_frames):
    """The persisted binding cannot certify its own object inventory."""
    world_ids = set(scene.actor_ids)
    for row in binding["frames"]:
        frame = raw_frames[row["navsim_index"]]
        if (
            frame["token"] != row["token"]
            or int(frame["timestamp"]) != row["timestamp_us"]
            or frame["log_name"] != binding["log_name"]
        ):
            raise ValueError("Raw source frame identity differs")
        anns = frame["anns"]
        if len(anns["track_tokens"]) != len(anns["gt_names"]) or len(
            set(anns["track_tokens"])
        ) != len(anns["track_tokens"]):
            raise ValueError("Invalid raw source inventory")
        expected = {
            str(t): (str(k), str(t) in world_ids)
            for t, k in zip(anns["track_tokens"], anns["gt_names"])
        }
        actual = {o["track_token"]: (o["kind"], o["physics_bound"]) for o in row["source_objects"]}
        if expected != actual or len(actual) != len(row["source_objects"]):
            raise ValueError("Binding object inventory differs from immutable raw annotations")


def verified_asset_preflight(binding, nodes, scene):
    import pickle
    from pathlib import Path

    from driversi.agents.backends.native import digest

    identity = binding["inputs"]["navsim_log"]
    path = Path(identity["path"])
    if digest(path) != identity["sha256"]:
        raise ValueError("Raw annotation file hash differs")
    with path.open("rb") as stream:
        raw_frames = pickle.load(stream)  # caller already explicitly trusts source assets
    verify_source_inventory(binding, scene, raw_frames)
    report = audit_binding_assets(binding, nodes)
    report["raw_annotation_sha256"] = identity["sha256"]
    report["inventory_verified_against_raw_log"] = True
    return report
