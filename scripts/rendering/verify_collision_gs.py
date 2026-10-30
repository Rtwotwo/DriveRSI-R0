"""Bounded real-asset regression for authoritative fleet presence and GS bindings."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from driversi.rendering.gaussian.mtgs import _array
from driversi.rendering.gaussian.mtgs_live_viewer import write_session_viewer
from driversi.rendering.gaussian.mtgs_runtime import MTGSRuntime, project_source_lens
from driversi.rendering.gaussian.mtgs_session import MTGSWorldSession


def fault_probe(runtime, session, output):
    """Test a labelled synthetic metadata fault using real pixels and rigid assets."""
    stamp = session.metadata["source_start_timestamp_us"] + round(
        session.current["tick"] * 0.1 * 1e6
    )
    _, _, _, camera, projection = session._sample("CAM_F0", stamp)
    frame = int(camera.metadata["frame_idx"])
    candidates = tuple(session.virtual_camera_actors)
    for actor in candidates:
        node = runtime.model.gaussian_models["rigid_object_" + actor]
        supported = np.flatnonzero(session._node_presence[actor][:frame])
        if not len(supported) or not bool(node.in_frame_mask[frame]):
            continue
        anchor = int(supported[-1])
        quat, trans = node.get_object_pose(frame, None)
        pose = (_array(trans).copy(), _array(quat).copy())
        poses = {actor: pose}
        raw_results = {}

        def render(label, **kwargs):
            result = runtime.render_camera(camera, **kwargs)
            raw_results[label] = result["rgb"]
            projected = project_source_lens(result, projection)
            return projected["rgb"], projected["valid"]

        reference, valid = render("reference", rigid_poses=poses)
        original_visible = camera.metadata.get("visible_tokens")
        try:
            # A world-owned car remains; simulate stale source annotation and culling.
            node.in_frame_mask[frame] = False
            camera.metadata["visible_tokens"] = [
                name for name in runtime.model.gaussian_models if name != "rigid_object_" + actor
            ]
            missing, _ = render("missing")
            fixed, _ = render("repaired", rigid_poses=poses, rigid_source_frames={actor: anchor})
            assert not bool(node.in_frame_mask[frame]), "Presence transaction leaked"
        finally:
            node.in_frame_mask[frame] = True
            if original_visible is None:
                camera.metadata.pop("visible_tokens", None)
            else:
                camera.metadata["visible_tokens"] = original_visible
        restored, _ = render("restored", rigid_poses=poses)
        changed = int(np.any(reference != missing, axis=-1).sum())
        if not changed:
            continue
        assert np.array_equal(reference, fixed), "Asset repair differs from matched reference"
        assert np.array_equal(reference, restored), "Fault changed subsequent rendering"
        for label, pixels in (
            ("reference", reference),
            ("missing", missing),
            ("repaired", fixed),
            ("restored", restored),
        ):
            Image.fromarray(pixels).save(output / ("fault-" + label + ".png"))
            Image.fromarray(raw_results[label]).save(output / ("fault-" + label + "-pinhole.png"))
        Image.fromarray(valid.astype(np.uint8)*255).save(output / "fault-valid.png")
        evidence = {
            "image_domain": "opencv_distorted_source_lens",
            "raw_pinhole_preserved": True,
            "lens_valid_pixels": int(valid.sum()),
            "kind": "explicit_metadata_fault_injection_on_real_asset",
            "actor_id": actor,
            "frame": frame,
            "anchor": anchor,
            "changed_pixels": changed,
            "repair_exact": True,
            "restore_exact": True,
            "scope": "runtime_presence_and_visibility_not_reconstruction_coverage",
        }
        return evidence
    raise AssertionError("No visible reconstructed vehicle exercised the injected fault")


def run(config_path, scene, output):
    cfg = json.loads(Path(config_path).read_text())
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    started = time.perf_counter()
    runtime = MTGSRuntime(
        **{k: Path(cfg[k]) for k in ("config_path", "checkpoint_path", "road_block_config")},
        scale=cfg.get("scale", 0.5),
        trusted_assets=True,
    )
    settings = {
        k: Path(cfg[k])
        for k in ("road_graph", "binding", "world_python", "native_path", "population_config")
    }
    settings.update(
        scene=Path(scene),
        cameras=tuple(cfg["cameras"]),
        sensor_stride=5,
        desired_speed=cfg["desired_speed"],
        height_mode=cfg["height_mode"],
        virtual_camera_actors=tuple(cfg["virtual_camera_actors"]),
    )
    from driversi.contracts.context import scene_fingerprint
    from driversi.contracts.scene import Scene
    from driversi.data.participants.fleet import configure_vehicle_fleet

    binding = json.loads(settings["binding"].read_text())

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    if binding["inputs"]["scene"]["sha256"] != sha(cfg["scene"]):
        raise ValueError("Parent GS binding differs from source scene")
    original = Scene.load(cfg["scene"])
    expected = configure_vehicle_fleet(original)
    if scene_fingerprint(expected) != scene_fingerprint(Scene.load(scene)):
        raise ValueError("Derived scene must equal the declared fleet conversion")
    binding["fleet_derivation"] = {
        "version": expected.metadata["vehicle_fleet"]["version"],
        "parent_binding_sha256": sha(settings["binding"]),
        "parent_scene_sha256": sha(cfg["scene"]),
        "conversion": "configure_vehicle_fleet",
        "exact_fingerprint_verified": True,
    }
    binding["inputs"]["scene"] = {"path": str(Path(scene).resolve()), "sha256": sha(scene)}
    derived_binding = output / "fleet-binding.json"
    derived_binding.write_text(json.dumps(binding, indent=2))
    settings["binding"] = derived_binding
    session = MTGSWorldSession(runtime=runtime, output=output / "session", **settings)
    try:
        initial = [
            np.asarray(Image.open(session.output / c["image"])).copy()
            for c in session.latest["cameras"]
        ]
        while not session.current["done"]:
            session.advance()
        session._publish()
        retained = [
            dict(tick=f["tick"], camera=c["camera_id"], **b)
            for f in session.manifest["frames"]
            for c in f["cameras"]
            for b in c.get("retained_asset_bindings", [])
        ]
        frames = len(session.manifest["frames"])
        final_tick = session.current["tick"]
        assert final_tick == len(session.times) - 1, "Incomplete real rollout"
        episode_manifest = session.output / "episode-manifest.json"
        episode_manifest.write_text(json.dumps(session.manifest, allow_nan=False))
        write_session_viewer(episode_manifest)
        injected = fault_probe(runtime, session, output)
        session.reset()
        exact = all(
            np.array_equal(a, np.asarray(Image.open(session.output / c["image"])))
            for a, c in zip(initial, session.latest["cameras"])
        )
        assert exact, "Reset changed initial images"
        # Replay the identical initial observation again to catch leaked GS masks/poses.
        for actor, mask in session._node_presence.items():
            np.testing.assert_array_equal(
                _array(runtime.model.gaussian_models["rigid_object_" + actor].in_frame_mask), mask
            )
        report = {
            "status": "completed",
            "tick": final_tick,
            "frames_before_reset": frames,
            "reset_rgb_exact": exact,
            "source_presence_restored": True,
            "retained_bindings": retained,
            "real_retention_case_exercised": bool(retained),
            "fault_injection": injected,
            "real_retention_limitation": None
            if retained
            else "this matched source window did not lose a bound vehicle asset",
            "recorded_episode": "session/episode-manifest.json",
            "retained_actor_ids": sorted({b["actor_id"] for b in retained}),
            "wall_seconds_including_load": time.perf_counter() - started,
            "trainable": False,
            "official_score": None,
            "scope": "real NAVSIM fleet lifecycle/render integration; no perceptual reconstruction quality claim",
        }
        (output / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.scene, args.output)
