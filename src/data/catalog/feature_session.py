"""Resident native encoder with checked, process-local provenance reuse."""

import copy
from pathlib import Path
from types import MappingProxyType

from driversi.agents.backends.native import digest
from driversi.data.catalog.store import Catalog

from . import features


def snapshot(config):
    model = Path(config["model_path"]).resolve(strict=True)
    paths = sorted(p for p in model.iterdir() if p.suffix in (".json", ".safetensors"))
    paths += [
        Path(features.__file__).with_name(n)
        for n in ("features.py", "vectors.py", "observations.py")
    ]
    result = {}
    for path in paths:
        stat = path.stat()
        result[str(path.resolve())] = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    return result


class FeatureSession:
    """One model/profile per worker. Pixel and admission checks remain per job."""

    def __init__(self):
        self.expected = None
        self.actual = None
        self.files = None
        self.backend = None
        self.encoder_calls = 0
        self.members_catalog = None
        self.members_state = None
        self.members_files = {}
        self.members = None

    def validate(self):
        if self.actual is not None and snapshot(self.actual["config"]) != self.files:
            raise ValueError("Resident encoder inputs changed; restart with verified inputs")

    def profile(self, expected):
        if self.actual is not None:
            if expected != self.expected:
                raise ValueError("A resident worker cannot mix encoder profiles")
            self.validate()
            return copy.deepcopy(self.actual)
        before = snapshot(expected["config"])
        actual = features.feature_profile(expected["config"])
        if snapshot(expected["config"]) != before:
            raise ValueError("Encoder inputs changed during profile verification")
        self.expected = copy.deepcopy(expected)
        self.actual = copy.deepcopy(actual)
        self.files = before
        return actual

    def encode(self, images):
        if self.actual is None:
            raise ValueError("Verify feature profile before using resident encoder")
        self.validate()
        reused = self.backend is not None
        if not reused:
            self.backend = features.VisionEncoder(self.actual["config"])
        values, cost = self.backend(images)
        self.validate()
        self.encoder_calls += 1
        return values, dict(
            cost, resident_encoder_reused=reused, resident_encoder_call=self.encoder_calls
        )

    def member_map(self, catalog):
        from driversi.rsi.experiments.navval_retrieval import training_members

        path = Path(catalog.path).resolve(strict=True)
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.members_catalog is not None:
            if Path(self.members_catalog.path).resolve() != path:
                raise ValueError("Resident worker cannot mix source catalogs")
            if self.members_state is not None and self.members_state[0] != identity:
                self.members_catalog.close()
                self.members_catalog = None
        if self.members_catalog is None:
            self.members_catalog = Catalog(path, "navtrain", readonly=True)
        cat = self.members_catalog
        try:
            version = cat.db.execute("PRAGMA data_version").fetchone()[0]
            state = (identity, version)
            same_files = all(digest(p) == sha for p, sha in self.members_files.items())
            if self.members is not None and state == self.members_state and same_files:
                return self.members
            from driversi.rsi.experiments.navval_retrieval import official_filter

            split_path, _ = official_filter("navtrain")
            before_files = {str(split_path): digest(split_path)}
            members, files = training_members(cat, "navtrain")
            after = path.stat()
            if (after.st_dev, after.st_ino) != identity or cat.db.execute(
                "PRAGMA data_version"
            ).fetchone()[0] != version:
                raise ValueError("Source catalog changed during full membership verification")
            if files != before_files or any(digest(p) != sha for p, sha in files.items()):
                raise ValueError("Official split changed during membership verification")
        except Exception:
            cat.close()
            self.members_catalog = None
            self.members_state = None
            self.members = None
            self.members_files = {}
            raise
        self.members = MappingProxyType(dict(members))
        self.members_files = files
        self.members_state = state
        return self.members
