import json

import numpy as np
import pytest
from tests.unit.rendering.test_visual_sampling import sampling_session

from driversi.agents.visual_control.image_moment import ImageMomentPolicy
from driversi.data.datasets.visual_samples import export_visual_samples
from driversi.sim.sensors.visual_sampling import ManagedVisualSampler

Image = pytest.importorskip("PIL.Image", reason="Actual PNG replay needs isolated renderer Pillow")


@pytest.fixture
def visual_source(tmp_path):
    root = tmp_path/"session"
    root.mkdir()
    session = sampling_session(root, encode=lambda array, path: Image.fromarray(array).save(path))
    sampler = ManagedVisualSampler(session, ("ego", "car"))
    for _ in range(6):
        sampler.advance()
    doc = session.manifest
    doc.update(status="completed", dt_seconds=.1, behavior_policies={a: {"policy_id": "visual", "version": ImageMomentPolicy.version}
                                                                for a in ("ego", "car")})
    path = root/"manifest.json"
    path.write_text(json.dumps(doc))
    return path, doc


def write_source(source):
    path, doc = source
    path.write_text(json.dumps(doc))
    (path.parent/"decisions.jsonl").write_text("".join(json.dumps(x)+"\n" for x in doc["decisions"]))
    return path


def test_visual_export_reconstructs_actual_pixels_actions_and_memory(visual_source, tmp_path):
    path, _ = visual_source
    result = export_visual_samples(path, tmp_path/"export")
    assert result["summary"]["decisions"] == 3
    assert result["summary"]["physics_steps"] == 6
    assert result["summary"]["actor_decisions"] == 6
    assert not result["trainable"] and not result["eligibility"]["visual_rl"]
    intervals = [json.loads(x) for x in (tmp_path/"export/decision-intervals.jsonl").read_text().splitlines()]
    assert [len(x["joint_actions"]) for x in intervals] == [2, 2, 2]
    assert [x["end_tick"] for x in intervals] == [2, 4, 6]
    assert not list((tmp_path/"export").glob("*.png"))


@pytest.mark.parametrize("mutation", [
    lambda d: d["decisions"][0]["actions"].update(ego=1),
    lambda d: d["decisions"][1]["actors"][0]["history_ticks"].__setitem__(0, 3),
    lambda d: d["decisions"][0]["identity"].update(session_id="other"),
    lambda d: d["decisions"][0]["actors"][0]["history"][0][0].update(camera_id="car__virtual_front"),
    lambda d: d["steps"][1]["policy_decision"].update(newly_decided=True),
    lambda d: d["steps"][2]["policy_actions"].update(car=7),
    lambda d: d["decisions"][0]["memory_after"][0].__setitem__(2, 99),
])
def test_visual_export_rejects_inconsistent_inputs_and_step_references(visual_source, tmp_path, mutation):
    mutation(visual_source[1])
    with pytest.raises(ValueError):
        export_visual_samples(write_source(visual_source), tmp_path/"rejected")
    assert not (tmp_path/"rejected").exists()


def test_changed_terminal_image_is_detected_even_if_never_used_for_a_decision(visual_source, tmp_path):
    path, doc = visual_source
    camera = doc["frames"][-1]["cameras"][0]
    Image.fromarray(np.full((4, 6, 3), 40, np.uint8)).save(path.parent/camera["image"])
    with pytest.raises(ValueError, match="Successor"):
        export_visual_samples(path, tmp_path/"bad-terminal")


def test_old_inspection_export_does_not_silently_misclassify_visual_data(visual_source, tmp_path):
    from driversi.data.assets.mtgs_episode import export_mtgs_episode

    with pytest.raises(ValueError):
        export_mtgs_episode(visual_source[0], tmp_path/"legacy")


def test_frozen_visual_export_remains_valid_after_live_append(visual_source, tmp_path):
    path, _ = visual_source
    with (path.parent/"decisions.jsonl").open("a") as stream:
        stream.write('{"later_generation": 1}\n{"incomplete_append":')
    result = export_visual_samples(path, tmp_path/"frozen")
    assert result["summary"]["decisions"] == 3
    assert json.loads((tmp_path/"frozen/initial-world.json").read_text())["0"]["tick"] == 0
