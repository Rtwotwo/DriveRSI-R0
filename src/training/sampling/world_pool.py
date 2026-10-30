"""Synchronous process shards over the same TeacherEnvironment, with bounded RPC waits.

Workers own native handles and never import a model or share a CUDA context.
Any worker failure closes the pool: a partially advanced batch is never returned.
"""
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass

from driversi.contracts.learning import JointDecision
from driversi.contracts.transition import Snapshot
from driversi.data.scenes.scene_store import SceneCache
from driversi.training.sampling.environment import TeacherEnvironment, validate_initial_world
from driversi.training.sampling.world_transport import DeadlineChannel


@dataclass
class WorldFrame:
    observation: object
    state: object
    motion: object
    max_steps: int
    presence: object = None


@dataclass
class WorldStep:
    rewards: object
    valid: object
    terminated: object
    truncated: object
    events: dict
    episode_done: bool
    on_policy_eligible: object = None


class WorldGroup:
    def __init__(self, assignments, progress_weight, cache_size, snapshots=None):
        self.envs = {}
        self.cache = SceneCache(cache_size)
        self.progress_weight = progress_weight
        try:
            for index, path in assignments:
                self.replace(index, path, None if snapshots is None else snapshots[index])
        except BaseException:
            self.close()
            raise

    def replace(self, index, path, snapshot=None):
        env = TeacherEnvironment(self.cache.get(path), self.progress_weight)
        try:
            validate_initial_world(env)
            if snapshot is not None:
                env.restore(Snapshot(**snapshot))
        except BaseException:
            env.close()
            raise
        if index in self.envs:
            self.envs[index].close()
        self.envs[index] = env
        return self.frame(index)

    def frame(self, index, observation=None):
        env = self.envs[index]
        return WorldFrame(env.observation() if observation is None else observation,
                          env.world.state(), env.world.motion(), env.world.max_steps, env.world.presence())

    def handle(self, operation, payload):
        if operation == "step":
            results = {}
            for index, decision in payload:
                result = self.envs[index].advance(decision)
                raw = result.transition
                step = WorldStep(result.rewards, raw.valid, raw.terminated, raw.truncated,
                                 raw.events, raw.episode_done, result.on_policy_eligible)
                results[index] = (step, self.frame(index, result.observation))
            return results
        if operation == "reset":
            return {index: self.replace(index, path) for index, path in payload}
        if operation == "snapshot":
            return {i: asdict(env.snapshot()) for i, env in self.envs.items()}
        if operation == "stats":
            return {"worlds": len(self.envs), "cached_scenes": len(self.cache.entries),
                    "scene_loads": self.cache.loads}
        raise ValueError(f"Unsupported world pool operation: {operation}")

    def close(self):
        for env in self.envs.values():
            env.close()
        self.envs.clear()


def _worker(connection, assignments, progress_weight, cache_size, snapshots):
    group = None
    sequence = 0
    try:
        group = WorldGroup(assignments, progress_weight, cache_size, snapshots)
        connection.send((0, True, {i: group.frame(i) for i, _ in assignments}))
        while True:
            sequence, operation, payload = connection.recv()
            if operation == "close":
                break
            connection.send((sequence, True, group.handle(operation, payload)))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as error:
        try:
            connection.send((sequence, False, f"{type(error).__name__}: {error}"))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if group is not None:
            group.close()
        connection.close()


class WorldPool:
    def __init__(self, paths, *, workers=0, progress_weight=.02, cache_size=32,
                 timeout=30., snapshots=None):
        if type(workers) is not int or not 0 <= workers <= len(paths):
            raise ValueError("workers must be between zero and the number of worlds")
        if not paths or not 0 < timeout < float("inf"):
            raise ValueError("World paths and a finite positive timeout required")
        self.workers = workers
        self.timeout = float(timeout)
        self.closed = False
        self.sequence = 0
        self.clients = []
        self.group = None
        self.current = {}
        self.count = len(paths)
        try:
            if not workers:
                self.group = WorldGroup(list(enumerate(paths)), progress_weight, cache_size, snapshots)
                self.current = {i: self.group.frame(i) for i in range(len(paths))}
            else:
                context = mp.get_context("spawn")
                for worker in range(workers):
                    assignments = [(i, paths[i]) for i in range(worker, len(paths), workers)]
                    parent, child = DeadlineChannel.pair()
                    shard_snapshots = None if snapshots is None else {i: snapshots[i] for i, _ in assignments}
                    process = context.Process(target=_worker,
                        args=(child, assignments, progress_weight, cache_size, shard_snapshots), daemon=True)
                    try:
                        process.start()
                    except BaseException:
                        parent.close()
                        child.close()
                        raise
                    child.close()
                    self.clients.append((parent, process, {i for i, _ in assignments}))
                for result in self._receive(self.clients, 0):
                    self.current.update(result)
        except BaseException:
            self.close()
            raise

    def _receive(self, clients, sequence, deadline=None):
        if deadline is None:
            deadline = time.monotonic() + self.timeout
        values = []
        for connection, process, _ids in clients:
            try:
                actual, ok, result = connection.recv(deadline)
            except TimeoutError as error:
                raise TimeoutError(f"World worker {process.pid} timed out receiving") from error
            if actual != sequence or not ok:
                raise RuntimeError(f"World worker {process.pid} failed: {result}")
            values.append(result)
        return values

    def request(self, operation, payload=None):
        if self.closed:
            raise RuntimeError("World pool is closed; restore a complete checkpoint to rebuild all worlds")
        try:
            if self.group is not None:
                return [self.group.handle(operation, payload)]
            deadline = time.monotonic() + self.timeout
            self.sequence += 1
            active = []
            for connection, process, ids in self.clients:
                subset = None if payload is None else [(i, p) for i, p in payload if i in ids]
                if payload is not None and not subset:
                    continue
                try:
                    connection.send((self.sequence, operation, subset), deadline)
                except TimeoutError as error:
                    raise TimeoutError(f"World worker {process.pid} timed out sending") from error
                active.append((connection, process, ids))
            return self._receive(active, self.sequence, deadline)
        except BaseException:
            self.close()
            raise

    def step(self, decisions):
        if self.closed:
            raise RuntimeError("World pool is closed; restore a complete checkpoint to rebuild all worlds")
        # Validate the entire synchronous batch before any shard can mutate. A
        # late/short batch is a caller error, not a partially committed world.
        if not isinstance(decisions, (list, tuple)) or len(decisions) != self.count:
            raise ValueError("Joint batch must contain exactly one decision per world")
        for index, decision in enumerate(decisions):
            observation = self.current[index].observation
            if (not isinstance(decision, JointDecision) or decision.key != observation.key
                    or decision.actor_ids != observation.actor_ids or observation.episode_done):
                raise ValueError(f"Joint batch has stale/foreign identity, actor order or ended world at {index}")
        try:
            merged = {}
            for result in self.request("step", list(enumerate(decisions))):
                if set(merged) & set(result):
                    raise RuntimeError("World workers returned duplicate world results")
                merged.update(result)
            if set(merged) != set(range(self.count)):
                raise RuntimeError("World workers returned an incomplete joint batch")
            frames = {i: merged[i][1] for i in range(self.count)}
            results = [merged[i][0] for i in range(self.count)]
            self.current.update(frames)
            return results
        except BaseException:
            # Once dispatched, any failure requires complete checkpoint recovery.
            self.close()
            raise

    def reset(self, replacements):
        if replacements:
            for result in self.request("reset", replacements):
                self.current.update(result)

    def snapshot(self):
        result = {}
        for shard in self.request("snapshot"):
            result.update(shard)
        return [result[i] for i in range(self.count)]

    def stats(self):
        return self.request("stats")

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.group is not None:
            self.group.close()
        for connection, process, _ids in self.clients:
            # Close only this pool's children, never unrelated server jobs.
            connection.close()
            if process.is_alive():
                process.terminate()
        for _connection, process, _ids in self.clients:
            process.join(timeout=1.)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.)
        self.current.clear()
