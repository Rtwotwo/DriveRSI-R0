"""Export a self-contained local gallery; no hosting or deployment integration."""
import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

from driversi.visualization.publication import gallery as publication

MAX_SITE_BYTES = 1_000_000_000
MAX_FILE_BYTES = 100_000_000
MAX_ARCHITECTURE_BYTES = 5_000_000
ARCHITECTURE_FILES = {
    "platform-progress.html": "index.html",
    "platform-progress.svg": "platform-progress.svg",
    "platform-progress.json": "platform-progress.json",
}
CARD_FIELDS = """id title subtitle city source_log scenario_id kind duration_seconds camera_count
frame_count trainable learned_self_play summary warnings data thumbnail source_type executed_branch
generator""".split()
ACTOR_FIELDS = """id x y z yaw speed length width kind controlled present alive collision offroad
role""".split()
CAMERA_FIELDS = """id owner_actor_id virtual image source comparison comparison_label valid_mask tick
time_seconds simulation_timestamp_us source_timestamp_us camera_timestamp_us source_camera_timestamp_us
capture_residual_us source_token warnings camera_to_world K distortion""".split()
SCORE_FIELDS = """label before after protocol official_benchmark score reference_score_in_branch_world
no_at_fault_collisions drivable_area_compliance ego_progress time_to_collision_within_bound comfort
driving_direction_compliance native_score_threshold score_pass scored_seconds actual_world_seconds
logged_benchmark trainable""".split()
REPORT_FIELDS = """source_log token arrival_tick history_ticks label_ticks teacher teacher_backend
arrival_method history_xy_deviation_m restore_exact qualified_teacher_branch charged_joint_steps""".split()
EVIDENCE_FIELDS = """physics_dt_seconds state_columns coordinate_frame join render_actor_rows_checked
image_policy score_protocol logged_benchmark trainable rendered_rgb_used_for_actions learned_self_play""".split()
CHECK_FIELDS = """single_episode timestamps_monotone world_render_ticks_equal camera_owner_present_each_frame
camera_id_owner_mapping_stable virtual_cameras_have_no_source_groundtruth media_hashes_verified
quality_certified new_rendering_performed""".split()
PERFORMANCE_FIELDS = """steps physics_seconds advance_wall_seconds initial_capture_seconds scope
step_latency_p50_seconds step_latency_p95_seconds advance_excluding_manifest_real_time_factor""".split()


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite value cannot be published")
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def redact_text(text):
    # Remove host access details even when they are embedded in explanatory text.
    text = re.sub(r"(?i)(?:https?|ssh|file)://[^\s<>\"']+", "[private link removed]", text)
    text = re.sub(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])(?::\d+)?\b", "[host removed]", text)
    text = re.sub(r"(?i)\b[A-Z]:[\\/][^\s<>\"']+", "[private path removed]", text)
    text = re.sub(r"(?<![\w/])/[A-Za-z0-9_.-]+(?:/[^\s<>\"']*)?", "[private path removed]", text)
    text = re.sub(r"(?i)(?:server|服务器)\s*32341|EABOT[_-]\d+", "静态展示", text)
    return text


def clean(value):
    finite_tree(value)
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    return value


def select(value, fields):
    if not isinstance(value, dict):
        raise ValueError("Expected a schema object")
    return {key: clean(value[key]) for key in fields if key in value}


def public_card(card):
    result = select(card, CARD_FIELDS)
    if "score" in card:
        result["score"] = select(card["score"], SCORE_FIELDS)
    if "camera_coverage" in card:
        result["camera_coverage"] = select(
            card["camera_coverage"], ("controlled_actors", "camera_owners", "configured_cameras")
        )
    if "playback" in card:
        result["playback"] = select(card["playback"], ("mode",))
        if "performance" in card["playback"]:
            result["playback"]["performance"] = select(
                card["playback"]["performance"], PERFORMANCE_FIELDS
            )
    return result


def numeric_points(value):
    if not isinstance(value, list):
        raise ValueError("Map geometry must be a numeric array")
    for item in value:
        if isinstance(item, list):
            numeric_points(item)
        elif type(item) not in (float, int) or not math.isfinite(item):
            raise ValueError("Map geometry must be finite numeric data")
    return value


def public_evidence(evidence):
    result = select(evidence, EVIDENCE_FIELDS)
    if "report_fields" in evidence:
        result["report_fields"] = select(evidence["report_fields"], REPORT_FIELDS)
    if "checks" in evidence:
        result["checks"] = select(evidence["checks"], CHECK_FIELDS)
    if "counts" in evidence:
        result["counts"] = select(evidence["counts"], (
            "controlled_actors", "roster", "virtual_owners", "frames", "rendered_rgb",
            "source_reference_rgb", "media_files",
        ))
    provenance = evidence.get("provenance", {})
    if provenance:
        result["provenance"] = select(provenance, ("manifest_sha256", "source_limits"))
    if "branches" in evidence:
        result["branches"] = {}
        for name in ("baseline", "teacher", "candidate", "reference"):
            if name not in evidence["branches"]:
                continue
            branch = evidence["branches"][name]
            exported = select(branch, ("completed_steps", "failure", "trajectory_admitted"))
            if "events" in branch:
                exported["events"] = select(branch["events"], (
                    "collision", "offroad", "goal", "route_support_end",
                ))
            if "score" in branch:
                exported["score"] = (select(branch["score"], SCORE_FIELDS)
                                     if branch["score"] is not None else None)
            if "object_contacts" in branch:
                exported["object_contacts"] = [select(contact, ("tick", "actors"))
                                                for contact in branch["object_contacts"]]
            result["branches"][name] = exported
    return result


def public_case(data):
    finite_tree(data)
    result = {"id": data["id"], "metadata": public_card(data["metadata"]),
              "roads": numeric_points(data.get("roads", [])), "frames": [],
              "evidence": public_evidence(data.get("evidence", {}))}
    for field in ("routes", "trajectories"):
        result[field] = []
        for item in data.get(field, []):
            exported = select(item, ("actor_id", "role"))
            exported["points"] = numeric_points(item["points"])
            result[field].append(exported)
    for frame in data["frames"]:
        exported = select(frame, ("tick", "time_seconds", "phase", "simulation_timestamp_us"))
        exported["actors"] = [select(actor, ACTOR_FIELDS) for actor in frame["actors"]]
        exported["cameras"] = [select(camera, CAMERA_FIELDS) for camera in frame.get("cameras", [])]
        result["frames"].append(exported)
    return result


def public_svg(raw):
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("SVG entity declarations are not supported")
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    svg = ET.fromstring(raw)
    for parent in list(svg.iter()):
        for child in list(parent):
            if child.tag.rsplit("}", 1)[-1] in ("metadata", "script", "foreignObject"):
                parent.remove(child)
        for key, value in list(parent.attrib.items()):
            local = key.rsplit("}", 1)[-1]
            if local.lower().startswith("on"):
                del parent.attrib[key]
            elif local in ("href", "src"):
                if not (value.startswith("#") or value.startswith("data:image/png;base64,")
                        or value.startswith("data:image/jpeg;base64,")):
                    del parent.attrib[key]
            elif "url(" in value and not re.fullmatch(r"url\(#[A-Za-z0-9_-]+\)", value):
                raise ValueError("SVG external style reference is not supported")
            else:
                parent.attrib[key] = redact_text(value)
        if parent.tag.rsplit("}", 1)[-1] == "style" and parent.text:
            if "@import" in parent.text or re.search(r"url\((?!#)", parent.text):
                raise ValueError("SVG external stylesheet references are not supported")
        if parent.text and parent.tag.rsplit("}", 1)[-1] != "style":
            parent.text = redact_text(parent.text)
        if parent.tail:
            parent.tail = redact_text(parent.tail)
    return ET.tostring(svg, encoding="utf-8", xml_declaration=True)


def public_html(raw):
    text = raw.decode("utf-8")
    text, replacements = re.subn(
        r'<p class="foot">.*?</p>',
        '<p class="foot">DriveRSI · 静态案例展示 · 预生成回放，非在线渲染；不改变世界、策略与训练记录</p>',
        text, flags=re.DOTALL,
    )
    if replacements != 1:
        raise ValueError("Expected exactly one known viewer footer")
    if re.search(r'(?:href|src)\s*=\s*[\"\'](?:/|https?:|file:)', text, re.I):
        raise ValueError("HTML contains non-relative resource references")
    text = text.replace("团队入口等待管理员配置", "使用本机启动的只读预览地址")
    if re.search(r"(?:8\.130\.174\.55|/data/xinyao|EABOT_|32341)", text):
        raise ValueError("HTML still contains server-specific details")
    return text.encode("utf-8")


def architecture_text(text):
    """Reject deployment-specific details and resources outside the small bundle."""
    # SVG namespace declarations are identifiers, not fetched resources.
    text = text.replace("http://www.w3.org/2000/svg", "").replace(
        "http://www.w3.org/1999/xlink", ""
    )
    if (re.search(r"(?:https?|ssh|file|ftp|data|javascript):|(?<!:)//[A-Za-z0-9]", text, re.I)
            or re.search(r"(?i)\b[A-Z]:[\\/]|(?:^|[\s\"'=(])/(?:data|home|root|tmp|mnt|srv|etc|Users|var)/", text)
            or re.search(r"(?:8\.130\.174\.55|EABOT_|32341)", text)):
        raise ValueError("Architecture contains a remote resource or private path")


def architecture_style(text):
    if (re.search(r"@import|\\|expression\s*\(", text, re.I)
            or any(not re.fullmatch(r"\s*['\"]?#[A-Za-z0-9_.:-]+['\"]?\s*", value)
                   for value in re.findall(r"url\s*\((.*?)\)", text, re.I | re.S))):
        raise ValueError("Architecture contains an external style resource")


class ArchitectureHTML(HTMLParser):
    """A resource allowlist for the trusted, self-contained architecture viewer."""

    def __init__(self, case_ids):
        super().__init__(convert_charrefs=True)
        self.case_ids = case_ids
        self.in_style = False
        self.in_script = False
        self.script = []

    def reference(self, value):
        if (value in ("platform-progress.svg", "platform-progress.json", "../index.html")
                or re.fullmatch(r"#[A-Za-z0-9_.:=/-]*", value)):
            return
        match = re.fullmatch(r"\.\./index\.html\?case=([A-Za-z0-9_-]+)", value)
        if not match or match[1] not in self.case_ids:
            raise ValueError("Architecture contains a resource outside its allowlist")

    def handle_starttag(self, tag, attrs):
        if tag in ("base", "iframe", "frame", "frameset", "form"):
            raise ValueError("Architecture contains an unsupported active element")
        self.in_style = tag == "style" or self.in_style
        self.in_script = tag == "script" or self.in_script
        for key, value in attrs:
            if key.startswith("on") or key in ("srcset", "ping", "srcdoc"):
                raise ValueError("Architecture contains an unsupported resource attribute")
            if tag == "meta" and key == "http-equiv":
                raise ValueError("Architecture cannot redirect through metadata")
            if key == "style" and value:
                architecture_style(value)
            if key in ("href", "src", "data", "poster", "action", "formaction", "xlink:href"):
                if tag == "script" or value is None:
                    raise ValueError("Architecture external scripts are not supported")
                self.reference(value)
            if value and re.search(r"url\s*\(", value, re.I):
                architecture_style(value)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "style":
            self.in_style = False
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        if self.in_style:
            architecture_style(data)
        if self.in_script:
            self.script.append(data)

    def validate_script(self):
        script = "\n".join(self.script)
        # Local JSON is the viewer's only network operation; DOM interaction stays inline.
        if re.search(r"\b(?:XMLHttpRequest|WebSocket|EventSource|importScripts|Worker|SharedWorker|sendBeacon)\b|\bimport\s*(?:\(|[\"'{*])", script):
            raise ValueError("Architecture contains unsupported network JavaScript")
        without_local_fetch = re.sub(
            r"\bfetch\s*\(\s*(['\"])platform-progress\.json\1\s*\)", "", script
        )
        if re.search(r"\bfetch\b", without_local_fetch):
            raise ValueError("Architecture fetch must name its local JSON specification")


def architecture_assets(root, case_ids):
    paths = {name: publication.checked_path(root, "assets/figures/" + name, required=False)
             for name in ARCHITECTURE_FILES}
    present = [path.exists() for path in paths.values()]
    if not any(present):
        return {}
    if not all(present):
        raise ValueError("Architecture export requires all three source assets")
    result = {}
    for name, path in paths.items():
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Architecture source must be a regular file")
            raw = stream.read(MAX_ARCHITECTURE_BYTES + 1)
        if len(raw) > MAX_ARCHITECTURE_BYTES:
            raise ValueError("Architecture asset exceeds its publication size limit")
        text = raw.decode("utf-8")
        architecture_text(text)
        if name.endswith(".html"):
            parser = ArchitectureHTML(case_ids)
            parser.feed(text)
            parser.close()
            parser.validate_script()
        elif name.endswith(".svg"):
            if re.search(r"<!DOCTYPE|<!ENTITY|<\?(?!xml\s)", text, re.I):
                raise ValueError("Architecture SVG declarations and processing instructions are not supported")
            svg = ET.fromstring(raw)
            if svg.tag.rsplit("}", 1)[-1] != "svg":
                raise ValueError("Architecture diagram must be SVG")
            for node in svg.iter():
                if node.tag.rsplit("}", 1)[-1] in ("script", "foreignObject"):
                    raise ValueError("Architecture SVG cannot contain active content")
                for key, value in node.attrib.items():
                    key = key.rsplit("}", 1)[-1]
                    if key.lower().startswith("on"):
                        raise ValueError("Architecture SVG cannot contain event handlers")
                    if key == "base":
                        raise ValueError("Architecture SVG cannot override its base path")
                    if key in ("href", "src") and not re.fullmatch(r"#[A-Za-z0-9_.:-]+", value):
                        raise ValueError("Architecture SVG references must be local fragments")
                    if key == "style" or re.search(r"url\s*\(", value, re.I):
                        architecture_style(value)
                if node.tag.rsplit("}", 1)[-1] == "style" and node.text:
                    architecture_style(node.text)
        else:
            def unique_keys(pairs):
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise ValueError("Duplicate architecture JSON key")
                    value[key] = item
                return value
            data = json.loads(text, object_pairs_hook=unique_keys)
            if not isinstance(data, dict):
                raise ValueError("Architecture specification must be a JSON object")
            finite_tree(data)

            def check_case_ids(value):
                if isinstance(value, dict):
                    if "case_ids" in value:
                        ids = value["case_ids"]
                        if (not isinstance(ids, list) or any(
                                not isinstance(cid, str) or cid not in case_ids for cid in ids)):
                            raise ValueError("Architecture JSON names an unknown gallery case")
                    for item in value.values():
                        check_case_ids(item)
                elif isinstance(value, list):
                    for item in value:
                        check_case_ids(item)

            check_case_ids(data)
            # Also inspect decoded strings so JSON escapes cannot hide resource paths.
            architecture_text(json.dumps(data, ensure_ascii=False))
        result["site/architecture/" + ARCHITECTURE_FILES[name]] = raw
    return result


def gallery_html(raw, include_architecture):
    text = public_html(raw).decode("utf-8")
    link = r'<a\b[^>]*\bhref\s*=\s*["\']architecture/index\.html["\'][^>]*>.*?</a>'
    existing = re.search(link, text, re.I | re.S)
    if not include_architecture:
        text = re.sub(link, "", text, flags=re.I | re.S)
    elif not existing:
        anchor = ('<a href="architecture/index.html" style="color:#b4d6e8;'
                  'font-size:12px;white-space:nowrap">平台架构与进度</a>')
        text, count = re.subn(r'(<nav\b[^>]*class=["\']headnav["\'][^>]*>)',
                              lambda match: match[1] + anchor, text, count=1, flags=re.I)
        if count != 1:
            raise ValueError("Expected gallery header navigation for architecture link")
    return text.encode("utf-8")


def no_symlink_components(path):
    path = path.absolute()
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError("Output paths may not traverse symlinks")
    return path


def _rename_noreplace(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOSYS, "renameat2 unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(target)
        raise OSError(code, os.strerror(code), str(target))


def rename_new_directory(source, target):
    """Install a new directory, including on NAS without RENAME_NOREPLACE.

    The fallback first claims an absent target using exclusive mkdir. Cooperating
    exporters cannot acquire that name. A brief empty directory is visible before
    rename; this is for first publication, not live-site replacement. It cannot
    protect against an unrelated actor deleting/replacing our claim concurrently.
    """
    try:
        _rename_noreplace(source, target)
        return
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            raise
    os.mkdir(target, mode=0o700)  # Fails on every existing target, including empty directories.
    claimed = target.stat(follow_symlinks=False)
    identity = (claimed.st_dev, claimed.st_ino)
    try:
        current = target.stat(follow_symlinks=False)
        if (not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != identity):
            raise RuntimeError("Publication directory claim changed")
        os.rename(source, target)  # Atomic replacement of only our just-created empty claim.
    except BaseException:
        try:
            current = target.stat(follow_symlinks=False)
            if ((current.st_dev, current.st_ino) == identity
                    and stat.S_ISDIR(current.st_mode)):
                os.rmdir(target)  # Never recursively remove contents inserted by anyone else.
        except OSError:
            pass
        raise


class Package:
    def __init__(self, stage):
        self.stage = stage
        self.files = []
        self.site_bytes = 0

    def add(self, name, raw=None, source=None):
        target = self.stage / name
        if target.exists():
            raise ValueError("Duplicate package file")
        size = source.stat().st_size if source is not None else len(raw)
        if size >= MAX_FILE_BYTES:
            raise ValueError("File reaches the 100 MB publication limit")
        if name.startswith("site/") and self.site_bytes + size >= MAX_SITE_BYTES:
            raise ValueError("Site reaches the 1 GB publication limit")
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        count = 0
        with target.open("xb") as destination:
            if source is None:
                destination.write(raw)
                digest.update(raw)
                count = len(raw)
            else:
                fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise ValueError("Media must be a regular file")
                    while chunk := stream.read(1024 * 1024):
                        count += len(chunk)
                        if count >= MAX_FILE_BYTES:
                            raise ValueError("Media grew beyond publication limit")
                        destination.write(chunk)
                        digest.update(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if count != size:
            raise ValueError("Source media changed size during export")
        if name.startswith("site/"):
            self.site_bytes += count
        self.files.append({"path": name, "bytes": count, "sha256": digest.hexdigest()})


def export(root, output=None):
    root = no_symlink_components(Path(root)).resolve()
    viz = root / "viz"
    if viz.is_symlink() or not viz.is_dir():
        raise ValueError("Expected the validated viz directory")
    output = no_symlink_components(Path(output) if output else
                                   root / "outputs/publications/drive-rsi-pages")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if output == root or root.is_relative_to(output) or output.is_relative_to(viz):
        raise ValueError("Output must not overlap project inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    try:
        with publication.publication_lock(viz):
            catalog = publication.load_catalog(viz)
            assets = set()
            seen_paths = set()
            cards, cases = [], []
            for card in catalog["cases"]:
                publication.validate_case(viz, card, assets)
                if card["data"] in seen_paths:
                    raise ValueError("Different cases share one data file")
                seen_paths.add(card["data"])
                cards.append(public_card(card))
                cases.append(public_case(publication.read_json(
                    publication.checked_path(viz, card["data"])
                )))
                if card.get("thumbnail"):
                    assets.add(card["thumbnail"])
            architecture = architecture_assets(root, {card["id"] for card in cards})
            package = Package(stage)
            review_files = []
            if (viz / "camera-review").exists():
                from driversi.rendering.camera_review.publication import bundle_files
                review_files = bundle_files(viz / "camera-review")
            html = gallery_html(publication.checked_path(viz, "index.html").read_bytes(), bool(architecture))
            if review_files and b'id="camera-review-link"' not in html:
                html = html.replace(b'<nav class="headnav">', b'<nav class="headnav"><a id="camera-review-link" href="camera-review/" style="color:#b4d6e8">Camera-bound review</a>', 1)
            package.add("site/index.html", html)
            for name, source in review_files:
                package.add("site/camera-review/" + name, source=source)
            for name, raw in architecture.items():
                package.add(name, raw)
            package.add("site/preview.svg", public_svg(publication.checked_path(
                viz, "preview.svg"
            ).read_bytes()))
            package.add("site/data/cases.json", publication.json_bytes({
                "schema_version": 1, "presentation_scope": clean(catalog.get(
                    "presentation_scope", publication.SCOPE)), "cases": cards,
            }))
            for card, data in zip(cards, cases, strict=True):
                package.add("site/" + card["data"], publication.json_bytes(data))
            for asset in sorted(assets):
                package.add("site/" + asset, source=publication.checked_path(viz, asset))
            # Validate the sanitized copy using the same publication contracts.
            public_assets = set()
            for card in cards:
                publication.validate_case(stage / "site", card, public_assets)
            manifest = {
                "schema_version": 1, "kind": "static_pages_package", "uploaded": False,
                "case_count": len(cards), "media_file_count": len(assets),
                "rendered_image_count": sum(len(frame["cameras"]) for data in cases
                                            for frame in data["frames"]),
                "site_bytes": package.site_bytes, "file_count": len(package.files),
                "manifest_scope": "All package files except package-manifest.json itself",
                "installation_policy": "new target only; atomic no-replace rename or exclusive empty directory claim on NAS",
                "original_media_bytes_preserved": True, "trainable": False,
                "learned_self_play": False, "files": package.files,
            }
            (stage / "package-manifest.json").write_bytes(publication.json_bytes(manifest))
        rename_new_directory(stage, output)
    except BaseException:
        # This is only the fresh mkdtemp directory owned by this invocation.
        if stage.exists() and stage.parent == output.parent and stage.name.startswith("." + output.name):
            shutil.rmtree(stage)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = export(args.root, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
