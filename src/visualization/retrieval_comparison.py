"""Prepare loopback-only retrieval comparison assets outside the publication tree."""

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from driversi.data.catalog.scene_hierarchy import digest, save_json

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
METHODS = ("visual", "hybrid", "cluster")


def read(path):
    return json.loads(Path(path).read_text())


def select_examples(pairs, poses):
    """Two fixed illustrations per query map: median visual top1 and disagreement."""
    ranked = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        ranked[pair["query"]][pair["method"]].append(pair)
    for methods in ranked.values():
        for rows in methods.values():
            rows.sort(key=lambda r: r["rank"])
    selections = []
    cities = sorted({poses[q]["map"] for q in ranked})
    def value(row):
        return math.log1p(row["distance_m"]) if row["same_map"] else math.log1p(1e7)
    for city in cities:
        queries = sorted(q for q in ranked if poses[q]["map"] == city)
        ordered = sorted(queries, key=lambda q: (value(ranked[q]["visual"][0]), q))
        median = ordered[len(ordered) // 2]
        selections.append((median, "该城市视觉Top1地理距离排序的中位案例"))
        remaining = [q for q in queries if q != median]
        if remaining:
            contrast = max(remaining, key=lambda q: (abs(value(ranked[q]["cluster"][0]) - value(ranked[q]["visual"][0])), q))
            selections.append((contrast, "该城市聚类与视觉Top1地理距离差异最大的案例"))
    return selections, ranked


def run(project, result):
    project, result = Path(project).resolve(), Path(result).resolve()
    if not result.is_relative_to(project / "exp"):
        raise ValueError("Private assets must stay in the experiment tree")
    summary = read(result / "summary.json")
    if summary["status"] != "complete":
        raise ValueError("Complete geographic audit required")
    private = result / "private"
    poses = {r["token"]: r for r in read(private / "poses.json")["rows"]}
    pairs = read(private / "pairs.json")["rows"]
    hybrid_path = project / "exp/chainflow-restart-32000-v1/mapping/private-hybrid-mapping.json"
    if digest(hybrid_path) != summary["files"][str(hybrid_path)]:
        raise ValueError("Historical ranking changed")
    hybrid_rows = read(hybrid_path)["rows"]
    top_ties = sum(sum(c["fusion_score"] == row["union"][0]["fusion_score"] for c in row["union"]) > 1 for row in hybrid_rows)
    selections, ranked = select_examples(pairs, poses)
    qmeta = {r["token"]: r for r in read(project / "exp/chainflow9405-cluster-mapping-v1/private/query-mapping.json")["rows"]}
    members = {}
    for split in ("navtrain", "navtest"):
        for line in (project / "lib/cluster_caption/clusters" / split / "members.jsonl").open():
            row = json.loads(line)
            members[row["token"]] = row
    selected_tokens = set()
    for q, _ in selections:
        selected_tokens.add(q)
        for method in METHODS:
            rows = ranked[q][method]
            selected_tokens.add(rows[0]["candidate"])
            near = [r for r in rows if r["same_map"]]
            if near:
                selected_tokens.add(min(near, key=lambda r: (r["distance_m"], r["rank"]))["candidate"])
    groups = defaultdict(dict)
    for token in selected_tokens:
        member = members[token]
        groups[member["source_jsonl"]][member["source_line"]] = token
    captions = {}
    for source, line_tokens in groups.items():
        for i, line in enumerate((project / source).open(), 1):
            if i in line_tokens:
                cap = json.loads(line)["cluster_caption"]
                if cap["token"] != line_tokens[i]:
                    raise ValueError("Caption lineage changed")
                captions[line_tokens[i]] = cap
    view = private / "review"
    view.mkdir(exist_ok=True, mode=0o700)
    (view / "images").mkdir(exist_ok=True)
    proof, cache = {}, {}

    def display(token):
        if token in cache:
            return cache[token]
        pose, cap = poses[token], captions[token]
        images = {}
        for camera in CAMERAS:
            source = Path(pose["images"][camera])
            key = hashlib.sha256((token + camera).encode()).hexdigest()[:20]
            target = view / "images" / (key + ".jpg")
            with Image.open(source) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
                original_size = image.size
                image.thumbnail((960, 600))
                image.save(target, quality=90)
            images[camera] = "images/" + target.name
            proof[key] = {"source": str(source), "source_sha256": digest(source),
                          "source_size": original_size, "preview_sha256": digest(target),
                          "camera": camera, "token": token, "split": pose["split"],
                          "timestamp_us": pose["timestamp_us"], "anchor_index": pose["anchor_index"]}
        node = {"split": pose["split"], "token": token if pose["split"] == "navtrain" else None,
                "map": pose["map"], "xy": pose["xy"], "yaw": pose["yaw"], "images": images,
                "description": cap.get("scene_description_zh", ""), "tags": cap.get("retrieval_tags", []),
                "navigation_command": cap.get("navigation_command"), "speed_kmh": cap.get("anchor_speed_kmh"),
                "leaf": members[token]["leaf"], "map_labels": members[token].get("map_labels", [])}
        cache[token] = node
        return node

    examples, contacts = [], []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    for number, (query, reason) in enumerate(selections, 1):
        example = {"id": f"Q{number:02}", "selection": reason, "query": display(query),
                   "score": qmeta[query]["score"], "score_group": qmeta[query]["group"], "methods": {}}
        for method in METHODS:
            rows = ranked[query][method]
            near = [r for r in rows if r["same_map"]]
            closest = min(near, key=lambda r: (r["distance_m"], r["rank"])) if near else None
            def present(pair):
                return {**{k: pair[k] for k in ("rank", "same_map", "distance_m", "heading_difference_deg")}, "scene": display(pair["candidate"])} if pair else None
            example["methods"][method] = {"top1": present(rows[0]), "nearest16": present(closest)}
        board = Image.new("RGB", (1920, 1270), "#edf2f6")
        draw = ImageDraw.Draw(board)
        draw.text((12, 10), f"{example['id']} | {poses[query]['map']} | CURRENT ANCHOR ONLY | private geographic illustration", font=font, fill="#142b42")
        cards = [("NAVTEST QUERY", example["query"], "reference")]
        for method in METHODS:
            pair = example["methods"][method]["top1"]
            info = f"{pair['distance_m']:.1f} m / {pair['heading_difference_deg']:.1f} deg" if pair["same_map"] else "different map / distance N/A"
            cards.append((method.upper() + " TOP1", pair["scene"], info))
        for col, (label, scene, info) in enumerate(cards):
            x = col * 480
            draw.text((x + 10, 45), label, font=font, fill="#142b42")
            draw.text((x + 10, 70), info, font=font, fill="#344c60")
            for row, camera in enumerate(CAMERAS):
                y = 100 + row * 290
                with Image.open(view / scene["images"][camera]) as im:
                    thumb = ImageOps.contain(im.convert("RGB"), (466, 260))
                board.paste(thumb, (x + (480 - thumb.width) // 2, y + 25 + (260 - thumb.height) // 2))
                draw.text((x + 10, y), camera, font=font, fill="#344c60")
        contact = example["id"] + "-top1.jpg"
        board.save(view / contact, quality=93)
        example["contact_sheet"] = contact
        contacts.append(contact)
        examples.append(example)
    payload = {"schema_version": 1, "scope": "private_loopback_retrieval_review", "summary": summary,
               "ranking_audit": {"hybrid_top_score_ties": top_ties, "queries": len(hybrid_rows)},
               "selection_policy": "每城市2例：视觉Top1地理距离中位例＋聚类/视觉Top1距离差异最大例。说明性选择，不代表语义准确率。",
               "caption_status": "机器草稿，观察图片为准；当前导航指令不等于实际动作。",
               "examples": examples}
    save_json(view / "browser.json", payload)
    recall_path = result / "geographic-recall.json"
    if recall_path.exists():
        save_json(view / "geographic-recall.json", read(recall_path))
    save_json(private / "image-provenance.json", proof)
    save_json(result / "visualization-audit.json", {"examples": len(examples), "source_images": len(proof),
              "current_anchor_only": True, "camera_identity_checked": True, "publication": "none",
              "private_path": str(view), "contact_sheets": contacts, "module_sha256": digest(__file__)})
    for p in private.rglob("*"):
        if p.is_file():
            os.chmod(p, 0o600)
    return {"examples": len(examples), "images": len(proof), "output": str(view)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project, args.result), ensure_ascii=False))


if __name__ == "__main__":
    main()
