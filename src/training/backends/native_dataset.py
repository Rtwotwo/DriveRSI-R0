"""Content fingerprinting of native NAVSIM inputs without unpickling objects."""

import hashlib
import pickletools
from pathlib import Path


def fingerprint_inputs(data_root, log_names):
    from driversi.agents.backends.native import digest

    root = Path(data_root).resolve()
    sensor_root = root / "sensor_blobs/trainval"
    image_paths = set()
    aggregate = hashlib.sha256()
    missing = []
    for name in sorted(log_names):
        path = root / "navsim_logs/trainval" / (name + ".pkl")
        raw = path.read_bytes()
        aggregate.update((str(path) + "\0" + hashlib.sha256(raw).hexdigest() + "\n").encode())
        found = 0
        # Read pickle opcodes as data; never invoke REDUCE/GLOBAL or object constructors.
        for op, arg, _ in pickletools.genops(raw):
            if not isinstance(arg, str) or not arg.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image = (sensor_root / arg).resolve()
            if not image.is_relative_to(sensor_root.resolve()):
                raise ValueError("Sensor reference escapes the original dataset")
            if not set(image.parts) & {"CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"}:
                continue
            image_paths.add(image)
            found += 1
        if not found:
            raise ValueError("Unsupported NAVSIM log: no literal four-camera image references")
    for image in sorted(image_paths):
        if not image.is_file():
            missing.append(str(image))
            value = "MISSING"
        else:
            value = digest(image)
        aggregate.update((str(image) + "\0" + value + "\n").encode())
    return {
        "algorithm": "sha256-sorted-log-and-camera-content-v1",
        "sha256": aggregate.hexdigest(),
        "log_count": len(log_names),
        "image_count": len(image_paths),
        "ready": not missing,
        "missing_count": len(missing),
        "missing_examples": missing[:10],
        "scope": "all literal F0/L0/R0/B0 image references in admitted source logs",
    }
