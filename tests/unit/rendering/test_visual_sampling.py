import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from driversi.agents.visual_control.image_moment import ImageMomentPolicy
from driversi.contracts.sampling import ActorVisualInput
from driversi.rendering.gaussian.mtgs_session import MTGSWorldSession
from driversi.sim.sensors.visual_sampling import (
    ManagedVisualSampler,
    body_camera_transform,
    visual_population,
)


def sampling_session(tmp_path, encode=None):
    session = SimpleNamespace(output=tmp_path, generation=0, sensor_stride=2, session_id="fixture")
    session.current = {"tick": 0, "all_actor_ids": ["ego", "car"], "controlled_ids": ["ego", "car"],
        "state": [[0, 0, 0, 0, 2, 0, 0, 20, 0, 0, 0, 0, 0]]*2,
        "motion": [[0, 0, 0]]*2, "alive": [True, True], "done": False}
    session.manifest = {"schema_version": 1, "kind": "mtgs_world_session", "session_id": "fixture",
        "status": "running", "trainable": False, "display_only": True, "steps": [], "frames": [],
        "protocol": {}, "limits": [], "initial_world_by_generation": {"0": copy.deepcopy(session.current)}}

    def capture(brightness=(255, 0)):
        session.latest = {"tick": session.current["tick"], "generation": session.generation, "cameras": []}
        session._sensor_buffers = {}
        for actor, name, value, shape in zip(("ego", "car"), ("CAM_F0", "car__virtual_front"), brightness, ((4, 6), (6, 8))):
            rgb = np.full((*shape, 3), value, np.uint8)
            valid = np.ones(shape, bool)
            image = f"g{session.generation}-t{session.current['tick']}-{name}.png"
            mask = image.replace(".png", "-valid.png")
            if encode is None:
                (tmp_path/image).write_bytes(rgb.tobytes())
                (tmp_path/mask).write_bytes(valid.tobytes())
            else:
                encode(rgb, tmp_path/image)
                encode(valid, tmp_path/mask)
            session._sensor_buffers[name] = {"rgb": rgb, "valid": valid}
            session.latest["cameras"].append({"camera_id": name, "owner_actor_id": actor,
                "K": np.eye(3).tolist(), "distortion": [0]*5, "camera_to_body": np.eye(4).tolist(),
                "camera_model": "pinhole", "camera_timestamp_us": 1000000+session.current["tick"]*100000,
                "image": image, "valid": mask,
                "image_pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "valid_pixel_sha256": hashlib.sha256(valid.tobytes()).hexdigest()})
        session.manifest["frames"].append(copy.deepcopy(session.latest))
    session.capture = capture
    session.decision_identity = lambda: {"session_id": session.session_id, "generation": session.generation,
        "tick": session.current["tick"], "policy_versions": {a: {"policy_id": "visual", "version": ImageMomentPolicy.version}
        for a in ("ego", "car")}}

    def advance(overrides=None, *, external_actions, expected_tick, decision_identity, policy_decision):
        assert expected_tick == session.current["tick"] and decision_identity == session.decision_identity()
        session.current["tick"] += 1
        session.current["done"] = session.current["tick"] == 6
        step = {**copy.deepcopy(session.current), "generation": session.generation,
                "policy_actions": dict(external_actions), "executed_actions": {**external_actions, **(overrides or {})},
                "policy_decision": copy.deepcopy(policy_decision)}
        session.manifest["steps"].append(step)
        if session.current["tick"] % 2 == 0:
            capture()
        return step
    session.advance = advance
    capture()
    return session


def test_heterogeneous_owned_images_drive_actions_and_held_steps_are_not_new_observations(tmp_path):
    session = sampling_session(tmp_path)
    sampler = ManagedVisualSampler(session, ("ego", "car"))
    first = sampler.advance()
    assert first["policy_actions"] == {"ego": 7, "car": 1}
    sampler.advance()
    assert len(session.manifest["decisions"]) == 1
    assert not session.manifest["steps"][1]["policy_decision"]["newly_decided"]
    session.capture((0, 0))
    third = sampler.advance()
    assert third["policy_actions"] == {"ego": 1, "car": 1}
    record = session.manifest["decisions"][-1]
    assert record["actors"][0]["history_ticks"] == [0, 2]
    assert record["actors"][1]["history_ticks"] == [0, 2]
    assert record["memory_before"][0][0] != record["memory_before"][1][0]
    assert record["memory_after"][0][2] == record["memory_after"][1][2] == 2
    assert record["rng_before"][0] != record["rng_before"][1]
    assert len((tmp_path/"decisions.jsonl").read_text().splitlines()) == 2
    assert "neighbors" not in ActorVisualInput.__dataclass_fields__


def test_actor_history_is_immutable_and_reset_clears_held_memory(tmp_path):
    session = sampling_session(tmp_path)
    sampler = ManagedVisualSampler(session, ("ego", "car"))
    sampler.advance()
    camera = sampler.histories["ego"][0][1][0]
    with pytest.raises(ValueError):
        camera.rgb[0, 0, 0] = 0
    sampler.reset()
    assert all(not values for values in sampler.histories.values())
    assert sampler.held_actions is None and not sampler.memories.any()


def test_failed_inference_rolls_back_actor_state_before_physics(tmp_path):
    session = sampling_session(tmp_path)
    sampler = ManagedVisualSampler(session, ("ego", "car"))
    session._sensor_buffers["CAM_F0"]["valid"][:] = False
    with pytest.raises(ValueError):
        sampler.advance()
    assert session.current["tick"] == 0 and not session.manifest["decisions"]
    assert all(not values for values in sampler.histories.values())
    assert not sampler.memories.any()


def test_complete_decision_identity_rejects_previous_reset_and_other_policy_before_mutation():
    session = MTGSWorldSession.__new__(MTGSWorldSession)
    session.session_id, session.generation = "fixture", 0
    session.current = {"tick": 0, "behavior_policies": {"ego": {"policy_id": "visual", "version": "v1"}}}
    old = session.decision_identity()
    session.generation = 1
    with pytest.raises(ValueError, match="identity"):
        session.advance(external_actions={"ego": 7}, expected_tick=0, decision_identity=old)
    wrong = session.decision_identity()
    wrong["policy_versions"]["ego"]["version"] = "v2"
    with pytest.raises(ValueError, match="identity"):
        session.advance(external_actions={"ego": 7}, expected_tick=0, decision_identity=wrong)
    assert session.current["tick"] == 0


def test_population_selects_only_explicit_visual_actors():
    pop, ids = visual_population(("ego", "car", "other"), ("ego", "car"), ("car",), None, 8)
    assert ids == ("car",) and pop.external_actor_ids == ("car",)
    assert pop.behavior_policies["ego"]["controller"] == "route-idm-v2"
    with pytest.raises(ValueError):
        visual_population(("ego", "car"), ("ego",), ("car",), None, 8)


def test_optical_transform_is_composed_once_about_display_body():
    body = np.eye(4)
    body[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    body[:3, 3] = [10, 20, 3]
    rig = np.eye(4)
    rig[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    rig[:3, 3] = [1, 0, 2]
    cam = body @ rig @ np.diag([1., -1., -1., 1.])
    np.testing.assert_allclose(body_camera_transform(cam[:3], body), rig)


def test_missing_front_camera_is_rejected_before_gpu_loading(monkeypatch, tmp_path):
    import sys

    from driversi.rendering.gaussian.mtgs_session import run_matched_mtgs

    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=SimpleNamespace()))

    loaded = []
    monkeypatch.setattr("driversi.rendering.gaussian.mtgs_runtime.MTGSRuntime", lambda **kwargs: loaded.append(kwargs))
    monkeypatch.setattr("driversi.rendering.gaussian.mtgs_session.Scene.load", lambda _path:
                        SimpleNamespace(actor_ids=("ego", "car"), controlled=(True, True)))
    paths = {key: tmp_path/key for key in ("config_path", "checkpoint_path", "road_block_config", "scene",
                                         "road_graph", "binding", "world_python", "native_path", "output")}
    with pytest.raises(ValueError, match="camera owners"):
        run_matched_mtgs(**paths, visual_policy=True, visual_actors=("ego",), cameras=("CAM_R0",))
    assert not loaded


def test_ended_session_does_not_create_a_visual_decision():
    session = MTGSWorldSession.__new__(MTGSWorldSession)
    session.current, session.manifest = {"done": True}, {"status": "completed"}
    session.visual_sampler = SimpleNamespace(advance=lambda _override: pytest.fail("must not infer"))
    with pytest.raises(RuntimeError, match="Reset"):
        session.advance()


def test_sampler_restores_odd_tick_held_action_and_owns_state(tmp_path):
    session = sampling_session(tmp_path)
    original = ManagedVisualSampler(session, ("ego", "car"))
    original.advance()
    snapshot = original.snapshot()
    restored = ManagedVisualSampler(session, ("ego", "car"))
    restored.restore(snapshot)
    assert restored.observed_tick == 0 and session.current["tick"] == 1
    assert restored.held_actions == {"ego": 7, "car": 1}
    assert restored.memories.tolist() == original.memories.tolist()
    snapshot["memories"][:] = -5
    snapshot["histories"]["ego"][0][1][0]["rgb"][:] = 50
    assert restored.memories[0, 2] == 1
    assert restored.histories["ego"][0][1][0].rgb[0, 0, 0] == 255
    with pytest.raises(ValueError):
        restored.histories["ego"][0][1][0].rgb[:] = 12
    restored.advance()
    assert session.current["tick"] == 2
    assert not session.manifest["steps"][-1]["policy_decision"]["newly_decided"]


def test_policy_adapter_declares_version_and_is_frozen(tmp_path):
    class OtherPolicy(ImageMomentPolicy):
        version = "test-other-policy-v1"
    policy = OtherPolicy()
    population, _ = visual_population(("ego", "car"), ("ego", "car"), ("car",), None, 8, policy)
    assert population.behavior_policies["car"]["version"] == policy.version
    assert population.behavior_policies["car"]["policy_id"] == "managed-visual"
    session = sampling_session(tmp_path)
    sampler = ManagedVisualSampler(session, ("ego", "car"), policy=policy)
    sampler.advance()
    policy.version = "changed"
    with pytest.raises(ValueError, match="descriptor"):
        sampler.advance()
    assert session.current["tick"] == 1


def test_restored_rng_reproduces_actual_stochastic_policy(tmp_path):
    from driversi.contracts.visual import VisualPolicyResult
    class RandomPolicy(ImageMomentPolicy):
        version = "test-explicit-rng-v1"
        def infer(self, observations, memories, rngs):
            actions = np.asarray([r.integers(0, 12) for r in rngs], np.int32)
            return VisualPolicyResult(actions, memories + 1)
    session = sampling_session(tmp_path)
    sampler = ManagedVisualSampler(session, ("ego", "car"), policy=RandomPolicy())
    sampler.advance()
    saved = sampler.snapshot()
    expected = [r.integers(0, 1000000) for r in sampler.rngs]
    sampler.restore(saved)
    assert [r.integers(0, 1000000) for r in sampler.rngs] == expected


def test_bad_sampler_state_does_not_mutate_live_memory(tmp_path):
    session = sampling_session(tmp_path)
    sampler = ManagedVisualSampler(session, ("ego", "car"))
    sampler.advance()
    saved = sampler.snapshot()
    saved["rngs"][1] = {}
    with pytest.raises(ValueError):
        sampler.restore(saved)
    assert sampler.memories[:, 2].tolist() == [1, 1]
    assert sampler.observed_tick == 0
