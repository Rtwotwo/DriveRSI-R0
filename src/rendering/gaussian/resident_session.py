"""One-asset resident renderer used through the unchanged provider validations."""

import contextlib
import inspect
import os
import threading
import time
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.runtime.budget import write_json


class ResidentRenderSession:
    def __init__(self, bundle, gpu, *, image_writer=None):
        import torch

        from driversi.rendering.gaussian.gsplat_prebuilt import load as load_gsplat
        prebuilt=load_gsplat()

        from driversi.rendering.gaussian.source_session import SourceSession
        from driversi.rendering.gaussian.worldengine_batch import BatchedPortableRenderer
        from driversi.rendering.gaussian.worldengine_portable import PortableRenderer

        self.bundle, self.gpu = Path(bundle).resolve(), gpu
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
            raise ValueError("Resident renderer requires a dedicated physical GPU process")
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(0.35, 0)
        start = time.perf_counter()
        self.checkpoint = self.bundle / "background" / (self.bundle.name + ".ckpt")
        self.metadata = self.bundle / "video_scene_dict.pkl"
        self.code = [
            Path(inspect.getfile(x))
            for x in (PortableRenderer, BatchedPortableRenderer, SourceSession)
        ]
        self.code.extend(
            [
                Path(__file__),
                Path(__file__).with_name("image_store.py"),
                Path(__file__).with_name("worldengine_replay.py"),
                Path(__file__).with_name("gsplat_prebuilt.py"),
            ]
        )
        if prebuilt:self.code.append(prebuilt)
        self.frozen = {
            str(p): digest(p)
            for p in [
                self.checkpoint,
                self.metadata,
                self.bundle / "road_height_map/road_height_map.npy",
                self.bundle / "road_height_map/sim2.json",
                *self.code,
            ]
        }
        self.branch = BatchedPortableRenderer(
            self.checkpoint, metadata=self.metadata, terrain=self.bundle / "road_height_map"
        )
        canonical = BatchedPortableRenderer(self.checkpoint, metadata=self.metadata)
        self.source = SourceSession(
            canonical, self.checkpoint, self.metadata, self.code, image_writer=image_writer
        )
        self.writer = image_writer
        self.check_inputs()
        self.asset_seconds = time.perf_counter() - start

    def check_inputs(self):
        if any(digest(p) != sha for p, sha in self.frozen.items()):
            raise ValueError("Resident asset, terrain or producer code changed")

    def __call__(self, argv, output, seconds, *, environment=None):

        if environment.get("CUDA_VISIBLE_DEVICES") != str(self.gpu):
            raise ValueError("Resident render GPU changed")
        self.check_inputs()
        module = argv[argv.index("-m") + 1]
        args = dict(zip(argv[argv.index("-m") + 2 :: 2], argv[argv.index("-m") + 3 :: 2]))
        started = time.perf_counter()
        writer_start = len(self.writer.rows) if self.writer else 0
        # A killed in-process stage leaves no success receipt and is never retried
        # implicitly. The lane supervisor additionally bounds the whole process.
        watchdog = threading.Timer(seconds, lambda: os._exit(124))
        watchdog.daemon = True
        watchdog.start()
        try:
            return self._execute(argv, output, module, args, started, writer_start)
        finally:
            watchdog.cancel()

    def _execute(self, argv, output, module, args, started, writer_start):
        from driversi.rendering.gaussian.source_session import replay_adapter
        from driversi.rendering.gaussian.worldengine_replay import run as replay

        with (Path(output) / "worker.log").open("w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                if module == "driversi.rendering.gaussian.worldengine_replay":
                    if Path(args["--bundle"]).resolve() != self.bundle:
                        raise ValueError("Resident asset changed")
                    replay(
                        scene_path=Path(args["--scene"]),
                        rollout_path=Path(args["--rollout"]),
                        bundle=self.bundle,
                        raw_log=Path(args["--raw-log"]),
                        output=Path(args["--output"]),
                        future_points=int(args["--future-points"]),
                        width=int(args["--width"]),
                        renderer=replay_adapter(self.branch),
                        image_writer=self.writer,
                    )
                    hit = False
                elif module == "driversi.rendering.gaussian.worldengine_source":
                    from driversi.rendering.gaussian.worldengine_source import run as source_run

                    value = source_run(
                        Path(args["--render-report"]), Path(args["--output"]), session=self.source
                    )
                    hit = value["canonical_cache"]["hit"]
                else:
                    raise ValueError("Resident renderer accepts only replay/source operations")
        self.check_inputs()
        elapsed = time.perf_counter() - started
        write_json(
            Path(output) / "resident-timing.json",
            dict(
                seconds=elapsed,
                source_cache_hit=hit,
                asset_load_seconds=self.asset_seconds,
                module=module,
                png_rows=self.writer.rows[writer_start:] if self.writer else [],
            ),
        )
