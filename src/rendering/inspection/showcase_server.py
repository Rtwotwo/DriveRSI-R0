"""Bounded read-only team gallery service with identity-checked tmux management.

This stdlib server is for team previews behind SSH or a managed proxy. Tmux
survives SSH disconnects, not container restarts; run ``start`` after restart.
The server has no simulator, renderer, training or mutation endpoints.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import ipaddress
import json
import logging
import logging.handlers
import mimetypes
import os
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote_to_bytes, urlsplit

PROJECT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT / "scripts/rendering/serve_showcase.py"
DEFAULT_CONFIG = PROJECT / "src/config/visualization/showcase.json"
CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self'"
)


@dataclass(frozen=True)
class RuntimeIdentity:
    """Identity shared by processes in this boot and PID namespace, never by NAS."""

    hostname: str
    boot_id: str
    pid_namespace: str

    @property
    def key(self):
        label = re.sub(r"[^a-zA-Z0-9_-]", "-", self.hostname)[:80] or "host"
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:20]
        return f"{label}-{digest}"


def current_runtime_identity():
    # Hostname alone may be duplicated across containers, and PIDs are namespace-local.
    # boot_id prevents PID/inode reuse after a kernel restart from adopting old state.
    try:
        boot_id = str(uuid.UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip()))
        pid_namespace = os.readlink("/proc/self/ns/pid")
    except (OSError, ValueError) as exc:
        raise RuntimeError("Cannot establish this Linux container's runtime identity") from exc
    if not re.fullmatch(r"pid:\[\d+\]", pid_namespace):
        raise RuntimeError("Invalid PID namespace identity")
    return RuntimeIdentity(socket.gethostname(), boot_id, pid_namespace)


@dataclass(frozen=True)
class Settings:
    path: Path
    root: Path
    runtime: Path
    runtime_base: Path
    runtime_identity: RuntimeIdentity
    host: str
    port: int
    tmux_session: str
    restart_initial_seconds: float
    restart_max_seconds: float
    log_max_bytes: int
    log_backups: int


def load_settings(path=DEFAULT_CONFIG, *, project=PROJECT):
    project, path = Path(project).resolve(), Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported service configuration")
    root = (project / value.get("root", "viz")).resolve()
    if root != (project / "viz").resolve() or not root.is_dir():
        raise ValueError("Gallery root must be this project's existing viz directory")
    # runtime_dir is a shared base only. Each host/container gets its own state,
    # log rotation and supervisor lock below it; legacy flat state is never adopted.
    runtime_base = (project / value.get("runtime_dir", "exp/services/showcase")).resolve()
    if not runtime_base.is_relative_to(project / "exp/services"):
        raise ValueError("Runtime state must remain under project exp/services")
    identity = current_runtime_identity()
    runtime = (runtime_base / "hosts" / identity.key).resolve()
    if not runtime.is_relative_to(runtime_base):
        raise ValueError("Host runtime directory must remain under its runtime base")
    host = str(ipaddress.IPv4Address(value.get("host", "127.0.0.1")))
    port = value.get("port", 38515)
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Expected an unprivileged TCP port")
    session = value.get("tmux_session", "driversi-viz")
    if not re.fullmatch(r"driversi-viz(?:-[a-z0-9_-]+)?", session):
        raise ValueError("Expected a dedicated driversi-viz tmux session")
    initial, maximum = (
        float(value.get("restart_initial_seconds", 1)),
        float(value.get("restart_max_seconds", 30)),
    )
    if not 0.1 <= initial <= maximum <= 60:
        raise ValueError("Restart backoff must remain between 0.1 and 60 seconds")
    log_bytes, backups = value.get("log_max_bytes", 5242880), value.get("log_backups", 3)
    if type(log_bytes) is not int or not 1024 <= log_bytes <= 100 * 1024 * 1024:
        raise ValueError("Invalid rotating log limit")
    if type(backups) is not int or not 1 <= backups <= 10:
        raise ValueError("Invalid log retention")
    return Settings(
        path=path,
        root=root,
        runtime=runtime,
        runtime_base=runtime_base,
        runtime_identity=identity,
        host=host,
        port=port,
        tmux_session=session,
        restart_initial_seconds=initial,
        restart_max_seconds=maximum,
        log_max_bytes=log_bytes,
        log_backups=backups,
    )


def request_parts(target):
    if len(target) > 4096:
        raise ValueError("Request target too long")
    url = urlsplit(target)
    if url.scheme or url.netloc:
        raise ValueError("Absolute request targets are not accepted")
    path = unquote_to_bytes(url.path).decode("utf-8", errors="strict")
    if not path.startswith("/") or "\\" in path or any(ord(c) < 32 or ord(c) == 127 for c in path):
        raise ValueError("Invalid path")
    if path == "/":
        return ("index.html",)
    parts = path[1:].split("/")
    if any(not p or p.startswith(".") for p in parts):
        raise ValueError("Hidden or traversal path")
    if parts == ["healthz"]:
        return ("healthz",)
    if parts in (["index.html"], ["preview.svg"]):
        return tuple(parts)
    if len(parts) > 1 and parts[0] == "data" and Path(parts[-1]).suffix == ".json":
        return tuple(parts)
    if (
        len(parts) > 1
        and parts[0] == "media"
        and Path(parts[-1]).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"}
    ):
        return tuple(parts)
    raise FileNotFoundError("Not a published gallery route")


def open_asset(root, parts):
    """Resolve containment then open each component with no-follow dir fds."""
    root = Path(root).resolve(strict=True)
    target = root.joinpath(*parts).resolve(strict=True)
    if not target.is_relative_to(root):
        raise PermissionError("Asset leaves gallery")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise PermissionError("Only regular assets are served")
    return os.fdopen(fd, "rb"), info


def make_server(root, *, host="127.0.0.1", port=0, logger=None, instance="manual"):
    root = Path(root).resolve(strict=True)
    logger = logger or logging.getLogger("driversi.showcase")

    class Handler(BaseHTTPRequestHandler):
        server_version = "DriveRSI-Viz"
        sys_version = ""

        def log_message(self, _format, *_args):
            # Never log request targets: query strings may carry client values.
            return

        def log_request(self, code="-", size="-"):
            logger.info("request method=%s status=%s bytes=%s", self.command, code, size)

        def reply_headers(self, status, length, content_type, *, etag=None, media=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if status != 304:
                self.send_header("Content-Length", str(length))
            self.send_header(
                "Cache-Control", "private, no-cache" if media else "no-store, max-age=0"
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            self.send_header("Content-Security-Policy", CSP)
            if status == 405:
                self.send_header("Allow", "GET, HEAD")
            if etag is not None:
                self.send_header("ETag", etag)
            self.end_headers()

        def failure(self, status):
            body = json.dumps({"error": status}).encode()
            self.reply_headers(status, len(body), "application/json; charset=utf-8")
            if self.command != "HEAD":
                self.wfile.write(body)

        def read_only(self):
            self.failure(405)

        do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_TRACE = do_CONNECT = read_only

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            try:
                parts = request_parts(self.path)
                if parts == ("healthz",):
                    body = json.dumps(
                        {
                            "service": "driversi-viz",
                            "status": "ok",
                            "instance": instance,
                            "published": (root / "index.html").is_file(),
                        }
                    ).encode()
                    self.reply_headers(200, len(body), "application/json; charset=utf-8")
                    if self.command != "HEAD":
                        self.wfile.write(body)
                    return
                stream, info = open_asset(root, parts)
                with stream:
                    media = parts[0] == "media" or parts == ("preview.svg",)
                    stamp = f"{info.st_dev}:{info.st_ino}:{info.st_mtime_ns}:{info.st_size}"
                    etag = 'W/"' + hashlib.sha256(stamp.encode()).hexdigest()[:32] + '"'
                    if media and etag in [
                        value.strip() for value in self.headers.get("If-None-Match", "").split(",")
                    ]:
                        self.reply_headers(
                            304, 0, "application/octet-stream", etag=etag, media=True
                        )
                        return
                    content_type = mimetypes.guess_type(parts[-1])[0] or "application/octet-stream"
                    if content_type in ("application/json", "text/html"):
                        content_type += "; charset=utf-8"
                    self.reply_headers(
                        200, info.st_size, content_type, etag=etag if media else None, media=media
                    )
                    if self.command != "HEAD":
                        shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
            except (UnicodeError, ValueError, PermissionError):
                self.failure(403)
            except FileNotFoundError:
                self.failure(404)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.EACCES):
                    self.failure(403)
                elif exc.errno not in (errno.EPIPE, errno.ECONNRESET):
                    logger.exception("asset read failed")

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True
        request_queue_size = 32

        def __init__(self, *args, **kwargs):
            self.clients = threading.BoundedSemaphore(32)
            super().__init__(*args, **kwargs)

        def process_request(self, request, client_address):
            if not self.clients.acquire(blocking=False):
                self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except BaseException:
                self.clients.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self.clients.release()

        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(15)
            return connection, address

    return Server((host, port), Handler)


def setup_log(config, name):
    config.runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = hashlib.sha256(str(config.runtime).encode()).hexdigest()[:20]
    logger = logging.getLogger("driversi.showcase." + name + "." + key)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            config.runtime / (name + ".log"),
            maxBytes=config.log_max_bytes,
            backupCount=config.log_backups,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def process_start(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rindex(")") + 2 :].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, ValueError, IndexError):
        return None


def matches_process(pid, started, operation, instance, config_path):
    if type(pid) is not int or pid <= 1 or not started or process_start(pid) != str(started):
        return False
    if not isinstance(instance, str) or not re.fullmatch(r"[0-9a-f]{32}", instance):
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
    except (OSError, UnicodeError):
        return False
    expected = [str(SCRIPT), operation, "--config", str(config_path), "--instance", instance]
    return command[-len(expected) :] == expected


def state_path(config):
    return config.runtime / "state.json"


def write_state(config, state):
    target = state_path(config)
    temporary = target.with_name("state.pending")
    state = {
        **state,
        "schema_version": 2,
        "runtime_identity": asdict(config.runtime_identity),
        "runtime_key": config.runtime_identity.key,
    }
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)


def read_state(config):
    try:
        state = json.loads(state_path(config).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(state, dict):
        return {}
    if (
        state.get("schema_version") != 2
        or state.get("config_path") != str(config.path)
        or state.get("runtime_identity") != asdict(config.runtime_identity)
        or state.get("runtime_key") != config.runtime_identity.key
    ):
        return {}
    return state


def status(config):
    state = read_state(config)
    token = state.get("instance")
    supervisor = matches_process(
        state.get("supervisor_pid"), state.get("supervisor_start"), "supervise", token, config.path
    )
    child = matches_process(
        state.get("server_pid"), state.get("server_start"), "serve", token, config.path
    )
    return {
        "status": "running"
        if supervisor and child
        else "recovering"
        if supervisor
        else "orphaned_own_server"
        if child
        else "stopped",
        "supervisor_verified": supervisor,
        "server_verified": child,
        "host": config.host,
        "port": config.port,
        "tmux_session": config.tmux_session,
        "runtime_dir": str(config.runtime),
        "runtime_base": str(config.runtime_base),
        "runtime_identity": asdict(config.runtime_identity),
        "runtime_key": config.runtime_identity.key,
        "legacy_shared_state_present": (config.runtime_base / "state.json").is_file(),
        "scope": "survives_ssh_disconnect_not_container_restart",
        "state": state,
    }


def terminate_owned(config, pid, started, operation, token, *, grace=5):
    if current_runtime_identity() != config.runtime_identity:
        return False
    if not matches_process(pid, started, operation, token, config.path):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not matches_process(pid, started, operation, token, config.path):
            return True
        time.sleep(0.1)
    if matches_process(pid, started, operation, token, config.path):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return True


def stop(config):
    state = read_state(config)
    token = state.get("instance")
    supervisor = terminate_owned(
        config, state.get("supervisor_pid"), state.get("supervisor_start"), "supervise", token
    )
    child = terminate_owned(
        config, state.get("server_pid"), state.get("server_start"), "serve", token
    )
    return {
        "signalled_owned_supervisor": supervisor,
        "signalled_owned_server": child,
        **status(config),
    }


def start(config):
    current = status(config)
    if current["status"] != "stopped":
        return current
    if not shutil.which("tmux"):
        raise RuntimeError("tmux is required; foreground serve is also available")
    # Refuse an occupied port. Never replace or terminate an unrelated listener.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((config.host, config.port))
        except OSError as exc:
            raise RuntimeError("Configured port is in use; no process was stopped") from exc
    token = uuid.uuid4().hex
    command = [
        sys.executable,
        "-B",
        str(SCRIPT),
        "supervise",
        "--config",
        str(config.path),
        "--instance",
        token,
    ]
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", config.tmux_session, shlex.join(command)],
        check=True,
        capture_output=True,
        text=True,
    )
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = status(config)
        if result["supervisor_verified"]:
            return result
        time.sleep(0.1)
    raise RuntimeError("tmux launched but supervisor identity was not published")


def serve(config, instance):
    logger = setup_log(config, "server")
    server = make_server(
        config.root, host=config.host, port=config.port, logger=logger, instance=instance
    )

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    logger.info("service started bind=%s:%s instance=%s", config.host, config.port, instance)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        logger.info("service stopped instance=%s", instance)


def supervise(config, instance):
    logger = setup_log(config, "supervisor")
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    base = {
        "schema_version": 2,
        "instance": instance,
        "config_path": str(config.path),
        "supervisor_pid": os.getpid(),
        "supervisor_start": process_start(os.getpid()),
        "server_pid": None,
        "server_start": None,
    }
    child, delay = None, config.restart_initial_seconds
    with (config.runtime / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while not stop_event.is_set():
                started = time.monotonic()
                command = [
                    sys.executable,
                    "-B",
                    str(SCRIPT),
                    "serve",
                    "--config",
                    str(config.path),
                    "--instance",
                    instance,
                ]
                child = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                base.update(
                    server_pid=child.pid,
                    server_start=process_start(child.pid),
                    phase="running",
                    restart_delay_seconds=delay,
                )
                write_state(config, base)
                logger.info("child started pid=%s instance=%s", child.pid, instance)
                while child.poll() is None and not stop_event.wait(0.2):
                    pass
                if stop_event.is_set():
                    terminate_owned(config, child.pid, base["server_start"], "serve", instance)
                    child.wait(timeout=2)
                    break
                code = child.wait()
                base.update(
                    server_pid=None, server_start=None, phase="backoff", last_exit_code=code
                )
                write_state(config, base)
                logger.warning("child exited code=%s; retry_seconds=%s", code, delay)
                if time.monotonic() - started >= 60:
                    delay = config.restart_initial_seconds
                if stop_event.wait(delay):
                    break
                delay = min(config.restart_max_seconds, delay * 2)
        finally:
            if child is not None and child.poll() is None:
                terminate_owned(config, child.pid, base["server_start"], "serve", instance)
                child.wait(timeout=2)
            base.update(server_pid=None, server_start=None, phase="stopped")
            write_state(config, base)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("start", "status", "stop", "serve", "supervise"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--instance", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = load_settings(args.config)
    if args.instance is not None and not re.fullmatch(r"[0-9a-f]{32}", args.instance):
        parser.error("Invalid service instance")
    if args.operation in ("serve", "supervise"):
        if args.operation == "supervise" and args.instance is None:
            parser.error("supervise is managed by start")
        function = serve if args.operation == "serve" else supervise
        function(config, args.instance or uuid.uuid4().hex)
    else:
        result = {"start": start, "status": status, "stop": stop}[args.operation](config)
        print(json.dumps(result, indent=2))
