"""Generated scene classification, isolated from both original scene libraries."""

import fcntl
import hashlib
import io
import json
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.data.catalog.store import identity
from driversi.runtime.budget import write_json


def library_root():
    return project_root() / "lib/cluster_caption/synthetic"


def enqueue(export_path, sample_report=None):
    """Durable local work item; never sends an API request or grants admission."""
    path = Path(export_path).resolve(strict=True)
    export = json.loads(path.read_text())
    if not str(export.get("token", "")).startswith("synthetic_"):
        raise ValueError("Generated token required; originals cannot enter synthetic library")
    root = library_root()
    key = identity(dict(export=str(path), sha256=digest(path)))
    directory = root / "jobs" / key
    directory.mkdir(parents=True, exist_ok=True)
    value = dict(
        schema_version=1,
        id=key,
        export=str(path),
        export_sha256=digest(path),
        token=export["token"],
        split="synthetic_navtrain",
        source_split="navtrain",
    )
    with (directory / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        job_path = directory / "job.json"
        if job_path.exists() and json.loads(job_path.read_text()) != value:
            raise ValueError("Synthetic job identity changed")
        write_json(job_path, value)
        if sample_report:
            sample = Path(sample_report).resolve(strict=True)
            write_json(
                directory / "admission-reference.json",
                dict(path=str(sample), sha256=digest(sample)),
            )
    return dict(id=key, job=str(job_path), status="queued", training_admission_granted=False)


def read_export(job):
    path = Path(job["export"])
    if digest(path) != job["export_sha256"]:
        raise ValueError("Synthetic export changed")
    export = json.loads(path.read_text())
    if export["token"] != job["token"]:
        raise ValueError("Synthetic token changed")
    from driversi.rsi.plugins.synthesis import check_files

    check_files(export["files"])
    return export


def prepare_anchor(job, output, policy):
    """Use executed anchor pixels/state; never the parent log's caption or future."""
    import numpy as np
    from PIL import Image

    from driversi.data.catalog.cluster_caption import CONFIG
    from driversi.data.sources.navsim import load_log
    from driversi.rendering.whitebox.conditioning import CAMERAS
    from driversi_agents.chainflow.synthetic import validate_source_train_membership

    export = read_export(job)
    frames = load_log(Path(export["log_path"]), trusted=True)
    anchor = export["anchor_index"]
    if anchor != 3 or len(frames) < 12 or frames[anchor]["token"] != export["token"]:
        raise ValueError("Expected executed four-frame history and native future window")
    times = [int(f["timestamp"]) for f in frames[:4]]
    if any(not 450000 <= b - a <= 550000 for a, b in zip(times, times[1:])):
        raise ValueError("Generated history is not continuous at 2 Hz")
    parent = export["parent"]
    if export["files"].get(str(Path(parent["log_path"]).resolve())) != digest(parent["log_path"]):
        raise ValueError("Original membership log is not sealed by generated export")
    original = load_log(Path(parent["log_path"]), trusted=True)
    metadata, tick = None, None
    if export.get("source_train_membership"):
        scene_paths = [p for p in export["files"] if Path(p).name == "scene.npz"]
        if len(scene_paths) != 1:
            raise ValueError("Unique sealed physical scene required")
        with np.load(scene_paths[0], allow_pickle=False) as scene:
            metadata = json.loads(str(scene["metadata"]))
        tick = (parent["anchor_index"] - metadata["source_start_frame"]) * 5
    validate_source_train_membership(
        parent, original, export.get("source_train_membership"), metadata=metadata, anchor_tick=tick
    )
    frame = frames[anchor]
    dynamics = np.asarray(frame["ego_dynamic_state"], dtype=float)
    if len(dynamics) < 4 or not np.isfinite(dynamics[:4]).all():
        raise ValueError("Finite generated anchor dynamics required")
    command = {
        (1, 0, 0, 0): "TURN LEFT",
        (0, 1, 0, 0): "GO FORWARD",
        (0, 0, 1, 0): "TURN RIGHT",
    }.get(tuple(frame["driving_command"]), "UNKNOWN")
    facts = dict(
        scope="current_anchor_only",
        future_included=False,
        anchor_speed_kmh=round(float(np.linalg.norm(dynamics[:2]) * 3.6), 2),
        navigation_command=command,
    )
    evidence = []
    root = Path(export["sensor_root"]).resolve(strict=True)
    for cam in CAMERAS:
        source = (root / frame["cams"][cam]["data_path"]).resolve(strict=True)
        if (
            not source.is_relative_to(root)
            or export["files"].get(str(source)) != digest(source)
            or str(source) not in export["image_files"]
        ):
            raise ValueError("Caption camera is not the sealed generated anchor")
        with Image.open(source) as im:
            im = im.convert("RGB")
            im.thumbnail((policy["image_max_side"], policy["image_max_side"]))
            stream = io.BytesIO()
            im.save(stream, format="JPEG", quality=policy["jpeg_quality"])
        data = stream.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        target = output / "images" / (sha + ".jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        if digest(target) != sha:
            raise ValueError("Compressed evidence changed")
        evidence.append(
            dict(
                id=cam + "_anchor",
                camera=cam,
                relative_time_s=0,
                source_timestamp_us=times[-1],
                source_file=str(source),
                source_sha256=digest(source),
                file=str(target),
                sha256=sha,
                bytes=len(data),
            )
        )
    profile = dict(
        prompt=(CONFIG / "cluster-caption-prompt.md").read_text(),
        facts=facts,
        evidence=evidence,
        source_log_sha256=digest(export["log_path"]),
    )
    request = dict(
        token=export["token"],
        split="synthetic_navtrain",
        scope="current_anchor_only",
        input_identity=identity(profile),
        **profile,
    )
    features = dict(
        anchor_speed_kmh=facts["anchor_speed_kmh"],
        anchor_ax_mps2=float(dynamics[2]),
        anchor_ay_mps2=float(dynamics[3]),
        nav_command_slot={"GO FORWARD": 0, "TURN LEFT": 1, "TURN RIGHT": 2}.get(command, 3),
    )
    position = np.asarray(frame["ego2global_translation"][:2], dtype=float)
    if not np.isfinite(position).all():
        raise ValueError("Finite world anchor required")
    member = dict(
        token=export["token"],
        source_log=parent["source_log"],
        history_continuous=True,
        features=features,
        map_labels=[],
        lane_association="unknown",
        map_query=dict(token=export["token"], map=frame["map_location"], xy=position.tolist()),
    )
    old = output / "caption-job.json"
    if old.exists() and json.loads(old.read_text()) != request:
        raise ValueError("Caption input changed; existing paid request must not be reused")
    write_json(old, request)
    return export, request, member


def topology(member, maps_root, output):
    from driversi.data.catalog.scene_topology import map_facts

    query = member["map_query"]
    paths = list((Path(maps_root) / query["map"]).glob("*/map.gpkg"))
    if len(paths) != 1:
        result = dict(
            status="unknown",
            reason="map_missing_or_ambiguous",
            features={},
            labels=[],
            lane_association="unknown",
            query=query,
        )
    else:
        result = dict(
            map_facts(paths[0], [query])[0],
            status="computed",
            query=query,
            map_file=str(paths[0]),
            map_sha256=digest(paths[0]),
        )
    write_json(output / "topology.json", result)
    return dict(
        member,
        features={**member["features"], **result["features"]},
        map_labels=result["labels"],
        lane_association=result["lane_association"],
    )


def assign(caption, member, model_root):
    """Transform only; never fit or modify original training/test memberships."""
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits

    from driversi.data.catalog.scene_hierarchy import (
        ancestor,
        assign_hierarchy,
        nearest_leaves,
        transform,
    )

    root = Path(model_root)
    names = ("representation.joblib", "hierarchy.json", "hierarchy.npz", "clusters.json")
    provenance = {name: digest(root / name) for name in names}
    tree = json.loads((root / "hierarchy.json").read_text())
    if tree["fit_split"] != "navtrain" or tree["config"].get("image_vectors_used") is not False:
        raise ValueError("Frozen training text/state/topology hierarchy required")
    model = joblib.load(root / "representation.joblib")
    if model["fit_split"] != "navtrain":
        raise ValueError("Training-fitted representation required")
    nodes = tree["nodes"]
    with np.load(root / "hierarchy.npz", allow_pickle=False) as arr:
        centers = arr["centers"]
        splits = dict(zip(arr["split_node_ids"].tolist(), arr["split_centers"]))
    with threadpool_limits(limits=2):
        x, _ = transform(dict(members=[member], captions=[caption]), model)
        assignment, _ = assign_hierarchy(x, nodes, splits)
        leaf = int(assignment[0])
        leaves = [n["id"] for n in nodes if not n["children"]]
        alt, dist = nearest_leaves(x, centers[leaves], leaves)
    summaries = json.loads((root / "clusters.json").read_text())["nodes"]
    node = next(n for n in summaries if n["id"] == leaf)
    distance = float(np.linalg.norm(x[0] - centers[leaf]))
    if provenance != {name: digest(root / name) for name in names}:
        raise ValueError("Hierarchy changed during assignment")
    return dict(
        leaf=leaf,
        levels={k: ancestor(leaf, nodes, d) for k, d in tree["config"]["cuts"].items()},
        label=node["label"],
        distance=distance,
        out_of_train_radius=distance > node["radius_p99"] + 1e-7,
        nearest_training_leaves=[
            dict(leaf=int(i), distance=float(d)) for i, d in zip(alt[0], dist[0])
        ],
        fit_split="navtrain",
        refitted=False,
        image_vectors_used=False,
        model_root=str(root),
        model_files=provenance,
    )


def admission(job, export, request):
    """Fail closed, but retain useful diagnostic captions in the quarantine pool."""
    from driversi.data.datasets.native_admission import validate_sample

    p = library_root() / "jobs" / job["id"] / "admission-reference.json"
    if not p.exists():
        return dict(valid=False, status="not_supplied")
    proof = json.loads(p.read_text())
    try:
        from driversi.data.datasets.synthetic_publish import dataset_root

        package = export.get("package_id")
        package_identity = export.get("package_identity")
        expected = hashlib.sha256(
            json.dumps(package_identity, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        canonical = dataset_root().resolve() / "samples" / str(package) / "export.json"
        if (
            not package_identity
            or package != expected
            or Path(job["export"]).resolve() != canonical
        ):
            raise ValueError("Diagnostic/benchmark export cannot enter qualified synthetic pool")
        if digest(proof["path"]) != proof["sha256"]:
            raise ValueError("Admission receipt changed")
        sample = validate_sample(proof["path"])
        if (
            sample["sample_id"] != export["token"]
            or sample["source_log"] != export["parent"]["source_log"]
        ):
            raise ValueError("Admission belongs to another scene")
        required = {
            str(Path(export["log_path"]).resolve()): digest(export["log_path"]),
            **{e["source_file"]: e["source_sha256"] for e in request["evidence"]},
        }
        if any(sample["files"].get(k) != v for k, v in required.items()):
            raise ValueError("Caption pixels/log differ from admitted native sample")
        return dict(valid=True, status="native_admission_validated", reference=proof)
    except (ValueError, KeyError, OSError) as exc:
        return dict(
            valid=False,
            status="native_admission_invalid",
            error_type=type(exc).__name__,
            reference=proof,
        )


def publish_record(value):
    """Atomic per-scene JSONL + full synthetic-only view; originals never opened."""
    root = library_root()
    directory = root / "records"
    directory.mkdir(parents=True, exist_ok=True)
    with (root / "publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = directory / (value["id"] + ".jsonl")
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        tmp.replace(p)
        rows = [json.loads(p.read_text()) for p in sorted(directory.glob("*.jsonl"))]
        tmp = root / "members.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        tmp.replace(root / "members.jsonl")
        pools = {}
        for row in rows:
            pools.setdefault(str(row["cluster"]["leaf"]), []).append(row["id"])
        write_json(
            root / "pools.json",
            dict(split="synthetic_navtrain", pools=pools, admission_recheck_required=True),
        )
        write_json(
            root / "manifest.json",
            dict(
                schema_version=1,
                split="synthetic_navtrain",
                members=len(rows),
                qualified_at_last_check=sum(r["native_admission"]["valid"] for r in rows),
                caption_quality="machine_drafts",
                original_libraries_modified=False,
                training_reader="driversi.data.catalog.synthetic_classification.training_members",
                training_admission_granted=False,
            ),
        )


def shared_transport(root, config, budget):
    """Called under api-budget.lock; rate/backoff/auth state survives each scene."""
    from driversi.data.catalog import bailian_caption as client

    path = root / "api-status.json"

    def send(body, key, private):
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get("authentication_blocked"):
            raise client.NotSent("shared_authentication_blocked_requires_operator")
        interval = max(
            60 / config["requests_per_minute"],
            60
            * (config["reserved_input_tokens"] + config["max_output_tokens"])
            / config["tokens_per_minute"],
        )
        earliest = max(state.get("last_started", 0) + interval, state.get("backoff_until", 0))
        time.sleep(max(0, earliest - time.time()))
        state["last_started"] = time.time()
        write_json(path, state)
        try:
            return budget(body, key, private)
        except client.APIError as exc:
            if exc.status in (401, 403):
                state["authentication_blocked"] = True
            if exc.status == 429:
                state["backoff_until"] = time.time() + 60
            write_json(path, state)
            raise

    return send


def process(job_path, config, *, transport=None):
    """One bounded and resumable scene. Existing received answers are reused."""
    from driversi.data.catalog import bailian_caption as client
    from driversi.data.catalog import cluster_caption as captions
    from driversi.data.catalog.cluster_caption_batch import SharedBudget

    job_path = Path(job_path).resolve(strict=True)
    job = json.loads(job_path.read_text())
    output = job_path.parent
    with (output / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        contract = output / "execution-policy.json"
        if contract.exists() and json.loads(contract.read_text()) != config:
            raise ValueError("Synthetic classification policy changed; retain prior request")
        write_json(contract, config)
        started = time.monotonic()
        export, request, member = prepare_anchor(job, output, config)
        root = library_root()
        # Serialize requests across workers: SharedBudget is thread-safe, not process-safe.
        with (root / "api-budget.lock").open("a") as budget_lock:
            fcntl.flock(budget_lock, fcntl.LOCK_EX)
            private = client.private_config(client.PRIVATE_ROOT / config["provider_config_name"])
            if private["models"] != [config["model"]]:
                raise ValueError("Unexpected caption model")
            private.update(
                concurrency=1,
                max_requests=1,
                max_attempts=1,
                max_tokens=config["max_output_tokens"],
                image_max_side=config["image_max_side"],
                jpeg_quality=config["jpeg_quality"],
                enable_thinking=False,
            )
            budget = SharedBudget(root, config)
            try:
                response = client.run(
                    [request],
                    private,
                    output / "api",
                    key=client.credential(client.PRIVATE_ROOT / config["credential_name"]),
                    transport=transport or shared_transport(root, config, budget),
                    decode=captions.decode,
                )
                cost = budget.snapshot()
            finally:
                budget.db.close()
        if not response["complete"]:
            report = dict(
                status="caption_incomplete",
                id=job["id"],
                response=response,
                cost=cost,
                training_admission_granted=False,
            )
            write_json(output / "report.json", report)
            return report
        receipt_path = Path(response["rows"][0]["file"])
        receipt = json.loads(receipt_path.read_text())
        caption = dict(
            schema_version=captions.VERSION,
            token=export["token"],
            split="synthetic_navtrain",
            status="machine_draft",
            observation_scope="current_anchor_four_cameras",
            future_images_included=False,
            navigation_command=request["facts"]["navigation_command"],
            anchor_speed_kmh=request["facts"]["anchor_speed_kmh"],
            **receipt["description"],
            model=receipt["model"],
            evidence_file=str(output / "caption-job.json"),
            receipt_file=str(receipt_path),
            review_status="unreviewed_machine_annotation",
            content_review_flags=receipt["content_review_flags"],
            native_prediction_target="unchanged_separate_native_export",
            indexable=False,
            trainable=False,
        )
        member = topology(member, config["maps_root"], output)
        cluster = assign(caption, member, project_root() / config["hierarchy"])
        proof = admission(job, export, request)
        value = dict(
            schema_version=1,
            id=job["id"],
            token=export["token"],
            split="synthetic_navtrain",
            source_split="navtrain",
            source_parent=export["parent"],
            source_train_membership=export.get("source_train_membership"),
            export=job["export"],
            export_sha256=job["export_sha256"],
            cluster_caption=caption,
            structured_state=member,
            cluster=cluster,
            native_admission=proof,
            partition="qualified" if proof["valid"] else "quarantine",
            training_admission_granted=False,
            native_training_reference=proof.get("reference"),
            evidence_files={
                str(p): digest(p)
                for p in (receipt_path, output / "caption-job.json", output / "topology.json")
            },
        )
        publish_record(value)
        report = dict(
            status="classified",
            id=job["id"],
            token=export["token"],
            partition=value["partition"],
            leaf=cluster["leaf"],
            out_of_train_radius=cluster["out_of_train_radius"],
            record=str(root / "records" / (job["id"] + ".jsonl")),
            cost=cost,
            requests_reused=response["reused"],
            elapsed_seconds=time.monotonic() - started,
            training_admission_granted=False,
        )
        write_json(output / "report.json", report)
        return report


def training_members(leaf=None):
    """Training consumers must revalidate native admission, not trust snapshot flags."""
    result = []
    for p in sorted((library_root() / "records").glob("*.jsonl")):
        row = json.loads(p.read_text())
        if leaf is not None and row["cluster"]["leaf"] != leaf:
            continue
        job = json.loads((library_root() / "jobs" / row["id"] / "job.json").read_text())
        export = read_export(job)
        if any(digest(k) != v for k, v in row["evidence_files"].items()):
            raise ValueError("Synthetic classification evidence changed")
        if any(
            digest(Path(row["cluster"]["model_root"]) / k) != v
            for k, v in row["cluster"]["model_files"].items()
        ):
            raise ValueError("Synthetic classification needs reassignment after model change")
        request = json.loads(Path(row["cluster_caption"]["evidence_file"]).read_text())
        proof = admission(job, export, request)
        if proof["valid"]:
            result.append(dict(row, native_admission=proof))
    return result
