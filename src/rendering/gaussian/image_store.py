"""Lossless PNG publication with optional local encoding and atomic NFS commit."""

import hashlib
import io
import os
import shutil
import tempfile
import time
from pathlib import Path


def flush_outputs(root):
    """Complete filesystem flushes for files owned by this output tree only."""
    root = Path(root).resolve()
    start, seen, count = time.perf_counter(), set(), 0
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir():
            directories.append(path)
        elif path.is_file() and path.resolve().is_relative_to(root):
            st = path.stat()
            identity = st.st_dev, st.st_ino
            if identity not in seen:
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                seen.add(identity)
                count += 1
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        fd = os.open(directory, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return dict(files=count, directories=len(directories), seconds=time.perf_counter() - start)


class PngWriter:
    def __init__(self, *, compression=1, staging=None):
        if type(compression) is not int or not 0 <= compression <= 9:
            raise ValueError("PNG compression must be0..9")
        self.compression = compression
        self.staging = Path(staging) if staging else None
        if self.staging:
            self.staging.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def __call__(self, pixels, destination):
        from PIL import Image

        destination = Path(destination)
        if destination.exists():
            raise ValueError("Refusing to replace existing rendered image")
        start = time.perf_counter()
        # One compressed frame in memory avoids NFS directory-removal races.
        # Publication remains lossless, fsynced and no-clobber on the destination FS.
        with io.BytesIO() as encoded:
            Image.fromarray(pixels).save(encoded, format="PNG", compress_level=self.compression)
            encode = time.perf_counter() - start
            expected = hashlib.sha256(encoded.getbuffer()).hexdigest()
            encoded.seek(0)
            fd, name = tempfile.mkstemp(prefix=".png-", dir=destination.parent)
            pending = Path(name)
            try:
                with os.fdopen(fd, "wb") as stream:
                    shutil.copyfileobj(encoded, stream, 1024 * 1024)
                    stream.flush()
                    os.fsync(stream.fileno())
                if hashlib.sha256(pending.read_bytes()).hexdigest() != expected:
                    raise ValueError("PNG publication bytes differ")
                # No-clobber atomic publication on the same filesystem.
                os.link(pending, destination)
                directory = os.open(destination.parent, os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                pending.unlink(missing_ok=True)
        self.rows.append(
            dict(
                bytes=destination.stat().st_size,
                encode_seconds=encode,
                total_seconds=time.perf_counter() - start,
            )
        )
