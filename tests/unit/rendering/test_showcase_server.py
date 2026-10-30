"""Real HTTP boundary checks; no production gallery or listener is touched."""

from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace

import pytest

from driversi.rendering.inspection import showcase_server as service


@pytest.fixture
def gallery(tmp_path):
    root = tmp_path / "viz"
    root.mkdir()
    (root / "data").mkdir()
    (root / "media").mkdir()
    (root / "index.html").write_text("<h1>Gallery</h1>")
    (root / "data/cases.json").write_text('{"cases":[]}')
    (root / "media/a.png").write_bytes(b"sample-image")
    (root / "preview.svg").write_text("<svg/>")
    (root / ".private.json").write_text("secret")
    outside = tmp_path / "private.json"
    outside.write_text("outside-secret")
    (root / "data/escape.json").symlink_to(outside)
    (root / "data/escape-dir").symlink_to(tmp_path, target_is_directory=True)
    (root / "data/inside.json").symlink_to(root / "data/cases.json")
    return root


@pytest.fixture
def endpoint(gallery):
    server = service.make_server(gallery, port=0, instance="test-instance")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(endpoint, target="/", method="GET", headers=None):
    connection = http.client.HTTPConnection(*endpoint, timeout=3)
    try:
        connection.request(method, target, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("path", ["/", "/index.html?scene=case&tick=30", "/data/cases.json"])
def test_current_pages_are_not_cached(endpoint, path):
    code, headers, body = request(endpoint, path)
    assert code == 200 and body
    assert headers["Cache-Control"] == "no-store, max-age=0"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert "object-src 'none'" in headers["Content-Security-Policy"]
    assert "ETag" not in headers


def test_head_and_health(endpoint):
    code, headers, body = request(endpoint, "/", "HEAD")
    assert code == 200 and body == b""
    assert int(headers["Content-Length"]) == len(b"<h1>Gallery</h1>")
    code, _, body = request(endpoint, "/healthz")
    assert code == 200
    assert json.loads(body) == {
        "service": "driversi-viz",
        "status": "ok",
        "instance": "test-instance",
        "published": True,
    }


def test_media_revalidates_and_changed_file_gets_new_etag(endpoint, gallery):
    code, headers, body = request(endpoint, "/media/a.png")
    assert code == 200 and body == b"sample-image"
    assert headers["Cache-Control"] == "private, no-cache"
    tag = headers["ETag"]
    code, cached_headers, body = request(endpoint, "/media/a.png", headers={"If-None-Match": tag})
    assert code == 304 and body == b""
    assert "Content-Length" not in cached_headers
    (gallery / "media/a.png").write_bytes(b"replacement-image")
    code, headers, body = request(endpoint, "/media/a.png", headers={"If-None-Match": tag})
    assert code == 200 and body == b"replacement-image" and headers["ETag"] != tag
    assert request(endpoint, "/preview.svg")[0] == 200


@pytest.mark.parametrize(
    "path",
    [
        "/../private.json",
        "/%2e%2e/private.json",
        "/data/../private.json",
        "/data/%2e%2e/private.json",
        "/.private.json",
        "/data/.private.json",
        "/data/a%5cprivate.json",
        "/data/%00.json",
        "/data/escape.json",
        "/data/escape-dir/private.json",
        "/data/inside.json",
    ],
)
def test_hidden_traversal_and_symlink_paths_are_denied(endpoint, path):
    code, _, body = request(endpoint, path)
    assert code == 403
    assert b"secret" not in body


@pytest.mark.parametrize(
    "path", ["/data", "/media", "/README.md", "/src/x.py", "/data/missing.json", "/media/script.js"]
)
def test_no_directory_listing_or_unpublished_routes(endpoint, path):
    assert request(endpoint, path)[0] == 404


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"])
def test_service_has_no_mutation_endpoints(endpoint, method):
    code, headers, _ = request(endpoint, "/", method)
    assert code == 405
    assert headers["Allow"] == "GET, HEAD"


def test_access_log_does_not_record_query(endpoint, caplog):
    with caplog.at_level(logging.INFO, logger="driversi.showcase"):
        assert request(endpoint, "/?private-token=do-not-log-me")[0] == 200
    assert "do-not-log-me" not in caplog.text
    assert "private-token" not in caplog.text
    assert "status=200" in caplog.text


@pytest.fixture
def settings(gallery, tmp_path):
    path = tmp_path / "service.json"
    path.write_text(json.dumps({"schema_version": 1}))
    return service.load_settings(path, project=tmp_path)


def test_configuration_is_project_scoped(settings, tmp_path):
    assert settings.root == tmp_path / "viz"
    assert settings.host == "127.0.0.1" and settings.port == 38515
    assert settings.runtime_base == tmp_path / "exp/services/showcase"
    assert settings.runtime == settings.runtime_base / "hosts" / settings.runtime_identity.key
    for change in (
        {"root": ".."},
        {"runtime_dir": "../../outside"},
        {"port": 22},
        {"host": "example.com"},
        {"tmux_session": "unrelated"},
    ):
        settings.path.write_text(json.dumps({"schema_version": 1, **change}))
        with pytest.raises(ValueError):
            service.load_settings(settings.path, project=tmp_path)


def test_process_identity_rejects_current_unrelated_python(settings):
    assert service.process_start(os.getpid())
    assert not service.matches_process(
        os.getpid(), service.process_start(os.getpid()), "serve", "a" * 32, settings.path
    )


def test_stop_does_not_signal_unrelated_pid(settings, monkeypatch):
    settings.runtime.mkdir(parents=True)
    service.write_state(
        settings,
        {
            "config_path": str(settings.path),
            "instance": "a" * 32,
            "supervisor_pid": os.getpid(),
            "supervisor_start": service.process_start(os.getpid()),
            "server_pid": os.getpid(),
            "server_start": service.process_start(os.getpid()),
        },
    )

    def unexpected_signal(*args):
        pytest.fail(f"Attempted unrelated process signal: {args}")

    monkeypatch.setattr(os, "kill", unexpected_signal)
    result = service.stop(settings)
    assert result["status"] == "stopped"
    assert not result["signalled_owned_supervisor"] and not result["signalled_owned_server"]


def test_start_refuses_occupied_port_without_launch_or_kill(settings, monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        settings = replace(settings, port=listener.getsockname()[1])
        monkeypatch.setattr(service.shutil, "which", lambda _name: "/usr/bin/tmux")

        def unexpected(*args, **kwargs):
            pytest.fail("Occupied port caused process operation")

        monkeypatch.setattr(service.subprocess, "run", unexpected)
        monkeypatch.setattr(os, "kill", unexpected)
        with pytest.raises(RuntimeError, match="no process was stopped"):
            service.start(settings)


def test_stale_state_cannot_be_adopted_by_other_config(settings):
    settings.runtime.mkdir(parents=True)
    service.write_state(settings, {"config_path": "/other/config.json", "instance": "a" * 32})
    assert service.read_state(settings) == {}
    assert service.status(settings)["scope"] == "survives_ssh_disconnect_not_container_restart"


def test_fifo_cannot_block_asset_reader(endpoint, gallery):
    os.mkfifo(gallery / "media/named-pipe.png")
    assert request(endpoint, "/media/named-pipe.png")[0] == 403


def test_owned_process_termination_and_start_time_identity(settings):
    token = "b" * 32
    command = [
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        str(service.SCRIPT),
        "serve",
        "--config",
        str(settings.path),
        "--instance",
        token,
    ]
    child = subprocess.Popen(command)
    try:
        started = service.process_start(child.pid)
        assert service.matches_process(child.pid, started, "serve", token, settings.path)
        assert not service.matches_process(
            child.pid, str(int(started) + 1), "serve", token, settings.path
        )
        assert not service.matches_process(child.pid, started, "serve", "c" * 32, settings.path)
        began = time.monotonic()
        assert service.terminate_owned(settings, child.pid, started, "serve", token, grace=1)
        child.wait(timeout=2)
        assert time.monotonic() - began < 2
        assert service.process_start(child.pid) is None
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=2)


def test_runtime_identity_stable_for_same_container():
    first, second = service.current_runtime_identity(), service.current_runtime_identity()
    assert first == second and first.key == second.key
    assert first.hostname and first.boot_id and first.pid_namespace.startswith("pid:[")


@pytest.mark.parametrize(
    "field,value",
    [
        ("hostname", "other-container"),
        ("boot_id", "00000000-0000-0000-0000-000000000001"),
        ("pid_namespace", "pid:[123]"),
    ],
)
def test_shared_nas_runtime_separates_host_boot_and_namespace(
    settings, tmp_path, monkeypatch, field, value
):
    foreign_identity = replace(settings.runtime_identity, **{field: value})
    monkeypatch.setattr(service, "current_runtime_identity", lambda: foreign_identity)
    other = service.load_settings(settings.path, project=tmp_path)
    assert settings.runtime_base == other.runtime_base
    assert settings.runtime != other.runtime
    assert service.state_path(settings) != service.state_path(other)
    assert settings.runtime / "supervisor.lock" != other.runtime / "supervisor.lock"
    settings.runtime.mkdir(parents=True)
    other.runtime.mkdir(parents=True)
    service.write_state(settings, {"config_path": str(settings.path), "instance": "a" * 32})
    service.write_state(other, {"config_path": str(other.path), "instance": "b" * 32})
    assert service.read_state(settings)["instance"] == "a" * 32
    assert service.read_state(other)["instance"] == "b" * 32


def test_foreign_state_in_local_directory_never_causes_process_lookup(settings, monkeypatch):
    settings.runtime.mkdir(parents=True)
    foreign = {
        "schema_version": 2,
        "config_path": str(settings.path),
        "instance": "a" * 32,
        "supervisor_pid": 1234,
        "supervisor_start": "111",
        "server_pid": 5678,
        "server_start": "222",
        "runtime_identity": asdict(
            replace(settings.runtime_identity, hostname="different-container")
        ),
        "runtime_key": settings.runtime_identity.key,
    }
    service.state_path(settings).write_text(json.dumps(foreign))

    def forbidden(*args, **kwargs):
        pytest.fail("Foreign state caused a process signal")

    monkeypatch.setattr(os, "kill", forbidden)
    assert service.read_state(settings) == {}
    result = service.stop(settings)
    assert result["status"] == "stopped"
    assert not result["signalled_owned_supervisor"] and not result["signalled_owned_server"]


def test_legacy_shared_state_is_reported_but_not_adopted(settings):
    settings.runtime_base.mkdir(parents=True)
    (settings.runtime_base / "state.json").write_text(
        json.dumps({"schema_version": 1, "config_path": str(settings.path), "instance": "a" * 32})
    )
    assert service.read_state(settings) == {}
    result = service.status(settings)
    assert result["legacy_shared_state_present"] and result["status"] == "stopped"


def test_identity_is_in_written_state(settings):
    settings.runtime.mkdir(parents=True)
    service.write_state(settings, {"config_path": str(settings.path), "instance": "a" * 32})
    state = service.read_state(settings)
    assert state["schema_version"] == 2
    assert state["runtime_identity"] == asdict(settings.runtime_identity)
    assert state["runtime_key"] == settings.runtime_identity.key


def test_termination_refuses_config_from_other_container(settings, monkeypatch):
    foreign = replace(
        settings, runtime_identity=replace(settings.runtime_identity, pid_namespace="pid:[987654]")
    )
    monkeypatch.setattr(service, "matches_process", lambda *args: True)
    monkeypatch.setattr(os, "kill", lambda *args: pytest.fail("Foreign config signalled a PID"))
    assert not service.terminate_owned(foreign, 1234, "started", "serve", "a" * 32)


def test_runtime_logs_do_not_share_handlers_between_containers(settings, tmp_path, monkeypatch):
    identity = replace(settings.runtime_identity, hostname="other-container")
    monkeypatch.setattr(service, "current_runtime_identity", lambda: identity)
    other = service.load_settings(settings.path, project=tmp_path)
    first = service.setup_log(settings, "test")
    second = service.setup_log(other, "test")
    try:
        first.info("local-only")
        second.info("foreign-only")
        assert "local-only" in (settings.runtime / "test.log").read_text()
        assert "foreign-only" not in (settings.runtime / "test.log").read_text()
        assert "foreign-only" in (other.runtime / "test.log").read_text()
        assert "local-only" not in (other.runtime / "test.log").read_text()
    finally:
        for logger in (first, second):
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
