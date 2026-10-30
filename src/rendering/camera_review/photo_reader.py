"""Resize trusted original photos in the existing image dependency environment."""

import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image

if __name__ == "__main__":
    request = json.load(sys.stdin)
    root = Path(request["output"])
    results = []
    for path, width, height in request["photos"]:
        with Image.open(path) as original:
            image = original.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        payload = buffer.getvalue()
        checksum = hashlib.sha256(payload).hexdigest()
        relative = f"media/{checksum}.png"
        (root / relative).write_bytes(payload)
        results.append(
            dict(
                key=[path, width, height],
                image=relative,
                raw_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                sha256=checksum,
            )
        )
    print(json.dumps(results))
