"""Opt-in feature/target cache reuse with on-demand, atomic cache writes.

SceneLoader remains responsible for split and invalid-token filtering. Cache
lookup uses each selected scene's log/token directly, without scanning other
sources or caching an entire split before DDP starts. Legacy Dataset behavior
is unchanged unless the training config selects cache_mode=read_through.
"""

import gzip
import logging
import os
import pickle
import tempfile
import zlib
from pathlib import Path

from navsim.planning.training.dataset import Dataset, dump_feature_target_to_pickle

logger = logging.getLogger(__name__)


def _atomic_dump(path: Path, data) -> None:
    """Publish a complete gzip on the same filesystem, including under DDP."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        dump_feature_target_to_pickle(temporary_path, data)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class ReadThroughDataset(Dataset):
    """Reuse existing builder caches and fill only missing samples on access."""

    def __init__(
        self,
        scene_loader,
        feature_builders,
        target_builders,
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=False,
    ):
        # Passing None deliberately avoids Dataset's directory scan and eager
        # cache_dataset(). All other feature/target augmentation stays intact.
        super().__init__(
            scene_loader=scene_loader,
            feature_builders=feature_builders,
            target_builders=target_builders,
            cache_path=None,
            force_cache_computation=False,
            append_token_to_batch=append_token_to_batch,
        )
        self._cache_path = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._refreshed_tokens = set()
        logger.info(
            "Read-through feature/target cache: root=%s samples=%d; "
            "existing files are reused, missing files are computed on access",
            self._cache_path,
            len(self),
        )

    def _token_cache_path(self, token: str) -> Path:
        frames = self._scene_loader.scene_frames_dicts[token]
        anchor = self._scene_loader._scene_filter.num_history_frames - 1
        return self._cache_path / frames[anchor]["log_name"] / token

    def _cache_scene_with_token(self, token: str, refresh: bool = False) -> None:
        token_path = self._token_cache_path(token)
        feature_builders = [
            builder for builder in self._feature_builders
            if refresh or not (token_path / f"{builder.get_unique_name()}.gz").is_file()
        ]
        target_builders = [
            builder for builder in self._target_builders
            if refresh or not (token_path / f"{builder.get_unique_name()}.gz").is_file()
        ]
        if feature_builders or target_builders:
            scene = self._scene_loader.get_scene_from_token(token)
            metadata = scene.scene_metadata
            if token_path != self._cache_path / metadata.log_name / metadata.initial_token:
                raise ValueError(f"Scene/cache identity mismatch for {token}: {token_path}")
            if feature_builders:
                agent_input = scene.get_agent_input()
                for builder in feature_builders:
                    _atomic_dump(
                        token_path / f"{builder.get_unique_name()}.gz",
                        builder.compute_features(agent_input),
                    )
            for builder in target_builders:
                _atomic_dump(
                    token_path / f"{builder.get_unique_name()}.gz",
                    builder.compute_targets(scene),
                )
        self._valid_cache_paths[token] = token_path

    def __getitem__(self, idx):
        if self._cache_path is None:
            return super().__getitem__(idx)

        token = self._scene_loader.tokens[idx]
        refresh = self._force_cache_computation and token not in self._refreshed_tokens
        self._cache_scene_with_token(token, refresh=refresh)
        if refresh:
            self._refreshed_tokens.add(token)
        try:
            sample = super().__getitem__(idx)
        except (FileNotFoundError, EOFError, gzip.BadGzipFile, pickle.UnpicklingError, zlib.error) as exc:
            logger.warning("Rebuilding unreadable feature/target cache for %s: %s", token, exc)
            self._cache_scene_with_token(token, refresh=True)
            sample = super().__getitem__(idx)

        cached_token = sample[1].get("token")
        if cached_token is not None and cached_token != token:
            raise ValueError(f"Cached target token {cached_token} does not match scene {token}")
        return sample
