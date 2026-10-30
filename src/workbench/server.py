"""Loopback-only, read-only HTTP service. No arbitrary filesystem routes."""
import argparse
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from driversi.workbench.catalog import SceneCatalog


def quality_summary(path):
    if path is None:
        return {"status": "unavailable", "reason": "no_report_registered", "components": []}
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Quality summary is too large")
    value = json.loads(path.read_text())
    if value.get("kind") != "platform_quality_report" or value.get("schema_version") != 1:
        raise ValueError("Unknown quality report")
    if not isinstance(value.get("generated_at"), str):
        raise ValueError("Quality evidence date must be an ISO timestamp")
    datetime.fromisoformat(value["generated_at"])
    allowed_metrics = {"focused_core_passed", "optional_sampling_passed", "final_core_passed",
        "optional_modules_skipped", "camera_frames", "diagnostic_usable", "high_quality_supervision",
        "high_quality_supervision_certified", "simulation_seconds", "advance_wall_seconds",
        "wall_clock_ratio", "first_startup_seconds_including_JIT", "official_scores_computed",
        "real_high_score_teacher_validated", "optional_suite_passed", "browser_passed", "model_adapters_added"}
    components = []
    for row in value["components"]:
        if (row.get("status") not in ("verified_scope", "limited", "not_verified", "blocked")
                or not isinstance(row.get("name"), str)
                or not isinstance(row.get("evidence"), str)
                or not isinstance(row.get("limits"), list)
                or any(not isinstance(x, str) for x in row["limits"])):
            raise ValueError("Invalid quality component")
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict) or not set(metrics) <= allowed_metrics or any(type(v) not in
                (int, float, bool, type(None)) for v in metrics.values()):
            raise ValueError("Quality metrics must be scalar summaries")
        components.append({k: row[k] for k in ("name", "status", "evidence", "limits")}
                          | {"metrics": metrics})
    return {"status": "evidence_snapshot", "generated_at": value["generated_at"],
            "industrial_ready": False, "components": components}


def make_handler(catalog, ledger=None, quality_report=None, tto=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _request_allowed(self):
            try:
                host = urlsplit("http://" + self.headers.get("Host", ""))
                if (host.hostname not in {"localhost", "127.0.0.1", "::1"}
                        or host.username or host.password or host.path or host.query
                        or host.fragment):
                    return False
                port = host.port or 80
                origin_text = self.headers.get("Origin")
                if origin_text:
                    origin = urlsplit(origin_text)
                    if (origin.scheme != "http" or origin.hostname != host.hostname
                            or (origin.port or 80) != port or origin.username or origin.password
                            or origin.path or origin.query or origin.fragment):
                        return False
                return self.headers.get("Sec-Fetch-Site") != "cross-site"
            except ValueError:
                return False

        def do_GET(self):
            if not self._request_allowed():
                self._send(403, b'{"error":"loopback same-origin requests only"}')
                return
            route = urlsplit(self.path)
            try:
                if route.path == "/api/health":
                    body = {"status": "ok", "read_only": True, "tto_available": tto is not None}
                elif route.path.startswith("/api/tto/"):
                    if tto is None:
                        raise ValueError("No trajectory audit registered")
                    protected = ledger.protected_groups() if ledger else ()
                    query = parse_qs(route.query, strict_parsing=True)
                    if any(len(v) != 1 for v in query.values()):
                        raise ValueError("Require unique query fields")
                    query = {k: v[0] for k, v in query.items()}
                    if route.path == "/api/tto/catalog" and not query:
                        body = tto.index(protected)
                    elif route.path == "/api/tto/scene" and set(query) == {"token"}:
                        body = tto.scene(query["token"], protected)
                    elif route.path == "/api/tto/candidate" and set(query) == {"token", "row"}:
                        body = tto.candidate(query["token"], int(query["row"]), protected)
                    elif route.path == "/api/tto/image" and set(query) == {"token", "frame", "camera"}:
                        data, content_type = tto.image(query["token"], int(query["frame"]), query["camera"], protected)
                        self._send(200, data, content_type)
                        return
                    else:
                        raise ValueError("Unknown trajectory inspection request")
                elif route.path == "/tto" and tto is not None:
                    self._send(200, Path(__file__).with_name("tto_viewer.html").read_bytes(), "text/html; charset=utf-8")
                    return
                elif route.path == "/api/quality":
                    if route.query:
                        raise ValueError("Quality report is selected by server CLI only")
                    body = quality_summary(quality_report)
                elif route.path == "/api/catalog":
                    body = catalog.index(ledger.protected_groups() if ledger else ())
                elif route.path == "/api/scene":
                    query = parse_qs(route.query)
                    if set(query) != {"id"} or len(query["id"]) != 1:
                        raise ValueError("Require one catalog scene id")
                    body = catalog.scene(query["id"][0], ledger.protected_groups() if ledger else ())
                elif route.path == "/api/harness":
                    if route.query:
                        raise ValueError("Harness tasks are selected by server ledger only")
                    from driversi.rsi.workflow.store import public_tasks
                    body = {"tasks": public_tasks(ledger) if ledger else [], "read_only": True}
                elif route.path == "/api/experiments":
                    body = {"experiments": [], "status": "no_ledger_registered"}
                    if ledger is not None:
                        body = ledger.public_summary()
                elif route.path == "/":
                    path = Path(__file__).with_name("viewer.html")
                    text = path.read_text() if path.exists() else (
                        "<h1>Scene workbench F0</h1>"
                        "<p>Reference-scene API ready; interactive view not installed.</p>"
                    )
                    self._send(200, text.encode(), "text/html; charset=utf-8")
                    return
                else:
                    self._send(404, b'{"error":"unknown route"}')
                    return
                self._send(200, json.dumps(body, ensure_ascii=False, allow_nan=False).encode())
            except (KeyError, ValueError, OSError):
                self._send(400, b'{"error":"invalid or unavailable catalog resource"}')

        def _send(self, code, data, content_type="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                             "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                             "frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(data)
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--port", type=int, default=38510)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--tto-audit", type=Path)
    parser.add_argument("--tto-dataset", type=Path)
    parser.add_argument("--tto-trusted-pickle", action="store_true")
    args = parser.parse_args(argv)
    catalog = SceneCatalog(args.suite)
    ledger = None
    if args.ledger:
        from driversi.rsi.execution.registry import ExperimentRegistry
        ledger = ExperimentRegistry(args.ledger, read_only=True)
    tto = None
    if args.tto_audit:
        if args.tto_dataset is None:
            parser.error("--tto-audit requires --tto-dataset")
        from driversi.workbench.tto_inspector import TTOInspector
        tto = TTOInspector(args.tto_audit, args.tto_dataset, trusted=args.tto_trusted_pickle)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(catalog, ledger, args.quality_report, tto))
    print(f"Scene workbench http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
