"""Loopback HTTP server for the ModelDebugger browser application."""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, unquote, urlsplit

from .config import DEFAULT_PORT
from .daytona import DaytonaError, delete_runtime, provision_runtime, recommend_gpu
from .debug_store import DebugStore, DebugStoreError
from .huggingface import InspectionError, inspect_account, inspect_model
from .http_client import HTTPResponse, http_request

COOKIE_NAME = "refusalscope_hf_session"
COOKIE_PATH = "/api/huggingface"
SESSION_SECONDS = 30 * 24 * 60 * 60
RUNTIME_COOKIE_NAME = "refusalscope_runtime_session"
RUNTIME_COOKIE_PATH = "/api/runtime"
RUNTIME_SESSION_SECONDS = 24 * 60 * 60
MAX_JSON_REQUEST_BYTES = 10 * 1024 * 1024
RUNTIME_TIMEOUT_SECONDS = 20 * 60
MINIMUM_WORKER_VERSION = (0, 4, 0)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_FILES = {
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/src/app.js": ("src/app.js", "text/javascript; charset=utf-8"),
    "/src/benchmark.js": ("src/benchmark.js", "text/javascript; charset=utf-8"),
    "/src/debugger.js": ("src/debugger.js", "text/javascript; charset=utf-8"),
    "/src/presentation.js": ("src/presentation.js", "text/javascript; charset=utf-8"),
    "/src/graph-routing.js": ("src/graph-routing.js", "text/javascript; charset=utf-8"),
    "/assets/tutorial/01-checkpoint-map.png": ("assets/tutorial/01-checkpoint-map.png", "image/png"),
    "/assets/tutorial/02-inference-profile.png": ("assets/tutorial/02-inference-profile.png", "image/png"),
    "/assets/tutorial/03-paired-comparison.png": ("assets/tutorial/03-paired-comparison.png", "image/png"),
    "/assets/tutorial/04-intervention-result.png": ("assets/tutorial/04-intervention-result.png", "image/png"),
    "/assets/tutorial/05-verification-result.png": ("assets/tutorial/05-verification-result.png", "image/png"),
}
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_.~\-]{1,2048}$")
MODEL_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)?$")
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9_./\-]+$")


@dataclass(slots=True)
class Session:
    token: str
    expires_at: float


@dataclass(slots=True)
class RuntimeSession:
    endpoint: str
    secret: str
    expires_at: float
    provider: str = "local"
    preview_token: str = ""
    sandbox_id: str = ""
    provider_key: str = ""
    gpu_type: str = ""
    quantization: str = "none"


class ApplicationState:
    def __init__(self, server_token: str = "", debug_database: str | Path = ":memory:") -> None:
        self.server_token = server_token
        self.debug_store = DebugStore(debug_database)
        self.local_worker_session = (
            Path(debug_database).parent / "local-worker.json"
            if str(debug_database) != ":memory:"
            else None
        )
        self.sessions: dict[str, Session] = {}
        self.runtime_sessions: dict[str, RuntimeSession] = {}
        self.public_cache: dict[str, bytes] = {}
        self.lock = threading.RLock()

    def close(self) -> None:
        with self.lock:
            runtimes = list(self.runtime_sessions.values())
            self.runtime_sessions.clear()
        for runtime in runtimes:
            self._cleanup_runtime(runtime)
        self.debug_store.close()

    def create_session(self, token: str) -> str:
        identifier = secrets.token_hex(32)
        now = time.time()
        with self.lock:
            self._remove_expired(now)
            self.sessions[identifier] = Session(token, now + SESSION_SECONDS)
        return identifier

    def session_token(self, identifier: str) -> str | None:
        now = time.time()
        with self.lock:
            self._remove_expired(now)
            session = self.sessions.get(identifier)
            return session.token if session else None

    def remove_session(self, identifier: str) -> None:
        with self.lock:
            self.sessions.pop(identifier, None)

    def create_runtime_session(
        self,
        endpoint: str,
        secret: str,
        *,
        provider: str = "local",
        preview_token: str = "",
        sandbox_id: str = "",
        provider_key: str = "",
        gpu_type: str = "",
        quantization: str = "none",
    ) -> str:
        identifier = secrets.token_hex(32)
        now = time.time()
        with self.lock:
            self._remove_expired(now)
            self.runtime_sessions[identifier] = RuntimeSession(
                endpoint=endpoint,
                secret=secret,
                expires_at=now + RUNTIME_SESSION_SECONDS,
                provider=provider,
                preview_token=preview_token,
                sandbox_id=sandbox_id,
                provider_key=provider_key,
                gpu_type=gpu_type,
                quantization=quantization,
            )
        return identifier

    def runtime_session(self, identifier: str) -> RuntimeSession | None:
        now = time.time()
        with self.lock:
            self._remove_expired(now)
            return self.runtime_sessions.get(identifier)

    def remove_runtime_session(self, identifier: str, *, cleanup: bool = False) -> RuntimeSession | None:
        with self.lock:
            runtime = self.runtime_sessions.pop(identifier, None)
        if cleanup and runtime is not None:
            self._cleanup_runtime(runtime)
        return runtime

    @staticmethod
    def _cleanup_runtime(runtime: RuntimeSession) -> str:
        if runtime.provider != "daytona" or not runtime.sandbox_id or not runtime.provider_key:
            return ""
        try:
            delete_runtime(runtime.provider_key, runtime.sandbox_id)
        except DaytonaError as error:
            return str(error)
        return ""

    def local_worker_secret(self, endpoint: str) -> str:
        path = self.local_worker_session
        if path is None:
            return ""
        try:
            if path.stat().st_size > 4096:
                return ""
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return ""
        secret = str(payload.get("secret", "")).strip() if isinstance(payload, dict) else ""
        recorded_endpoint = _runtime_endpoint(str(payload.get("endpoint", ""))) if isinstance(payload, dict) else ""
        return secret if recorded_endpoint == endpoint and TOKEN_PATTERN.fullmatch(secret) else ""

    def cache_get(self, key: str) -> bytes | None:
        with self.lock:
            return self.public_cache.get(key)

    def cache_put(self, key: str, body: bytes) -> None:
        with self.lock:
            self.public_cache[key] = body

    def _remove_expired(self, now: float) -> None:
        expired = [identifier for identifier, session in self.sessions.items() if session.expires_at <= now]
        for identifier in expired:
            del self.sessions[identifier]
        runtime_expired = [
            identifier
            for identifier, session in self.runtime_sessions.items()
            if session.expires_at <= now
        ]
        for identifier in runtime_expired:
            del self.runtime_sessions[identifier]


class RefusalScopeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: ApplicationState) -> None:
        self.state = state
        super().__init__(address, RefusalScopeHandler)


class RefusalScopeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ModelDebugger"
    sys_version = ""
    server: RefusalScopeServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        if path == "/api/health":
            self._send_json(200, {"backend": "python", "ok": True})
        elif path == "/api/huggingface/account":
            self._handle_account()
        elif path == "/api/huggingface":
            self._handle_model(parsed.query)
        elif path == "/api/runtime/status":
            self._handle_runtime_status()
        elif path == "/api/debug/cases":
            self._handle_debug_case_list()
        elif path in {"/api/debug/dataset", "/api/huggingface/dataset"}:
            self._handle_debug_dataset(parsed.query)
        elif _debug_case_id(path):
            self._handle_debug_case_get(_debug_case_id(path) or "")
        elif path == "/api/huggingface/logout":
            self._send_error(405, "Method not allowed")
        else:
            self._serve_static(path)

    def do_POST(self) -> None:
        path = unquote(urlsplit(self.path).path)
        if path == "/api/huggingface/logout":
            self._handle_logout()
        elif path == "/api/runtime/connect":
            self._handle_runtime_connect()
        elif path == "/api/runtime/daytona/recommend":
            self._handle_daytona_recommend()
        elif path == "/api/runtime/daytona/provision":
            self._handle_daytona_provision()
        elif path == "/api/runtime/disconnect":
            self._handle_runtime_disconnect()
        elif path in {
            "/api/runtime/load",
            "/api/runtime/forward",
            "/api/runtime/generate",
            "/api/runtime/compare",
            "/api/runtime/intervene",
            "/api/runtime/sweep",
            "/api/runtime/root-cause",
            "/api/runtime/verify",
            "/api/runtime/activation",
        }:
            self._handle_runtime_proxy(path.removeprefix("/api/runtime"))
        elif path == "/api/debug/cases":
            self._handle_debug_case_create()
        else:
            self._send_error(405, "Method not allowed")

    def do_HEAD(self) -> None:
        self._send_error(405, "Method not allowed")

    def do_PUT(self) -> None:
        path = unquote(urlsplit(self.path).path)
        case_id = _debug_case_id(path)
        if case_id:
            self._handle_debug_case_replace(case_id)
        else:
            self._send_error(405, "Method not allowed")

    def do_DELETE(self) -> None:
        path = unquote(urlsplit(self.path).path)
        case_id = _debug_case_id(path)
        if case_id:
            self._handle_debug_case_delete(case_id)
        else:
            self._send_error(405, "Method not allowed")

    def _cookie_value(self, name: str) -> str:
        for item in self.headers.get("Cookie", "").split(";"):
            key, separator, value = item.strip().partition("=")
            if separator and key == name:
                return value.strip()
        return ""

    def _user_token(self) -> tuple[str, bool, bool, bool]:
        """Return token, present, valid, and whether it came from the cookie."""
        authorization = self.headers.get("Authorization", "").strip()
        if authorization:
            scheme, separator, token = authorization.partition(" ")
            valid = bool(separator and scheme.lower() == "bearer" and TOKEN_PATTERN.fullmatch(token))
            return token if valid else "", True, valid, False
        session_id = self._cookie_value(COOKIE_NAME)
        if not session_id:
            return "", False, True, False
        if not TOKEN_PATTERN.fullmatch(session_id):
            return "", True, False, True
        token = self.server.state.session_token(session_id)
        return token or "", True, token is not None, True

    def _handle_account(self) -> None:
        token, present, valid, from_cookie = self._user_token()
        if not present:
            self._send_error(401, "Enter a Hugging Face access token")
            return
        if not valid:
            message = "The saved Hugging Face session was invalid" if from_cookie else "Use a valid Hugging Face Bearer token"
            self._send_error(400, message, expire_cookie=from_cookie)
            return
        try:
            account = inspect_account(token)
        except InspectionError as error:
            if error.status in {401, 403}:
                if from_cookie:
                    self._discard_request_session()
                message = (
                    "Hugging Face rejected this saved access token"
                    if from_cookie
                    else "Hugging Face rejected this access token"
                )
                self._send_error(401, message, expire_cookie=from_cookie)
            else:
                self._send_error(_api_error_status(error.status), str(error))
            return
        session_id = self._cookie_value(COOKIE_NAME) if from_cookie else self.server.state.create_session(token)
        self._send_json(200, account, set_cookie=_persistent_cookie(session_id))

    def _handle_logout(self) -> None:
        self._discard_request_session()
        self._send_json(200, {"ok": True}, set_cookie=_expired_cookie())

    def _handle_model(self, query: str) -> None:
        parameters = parse_qs(query, keep_blank_values=True)
        model_id = parameters.get("model", [""])[0].strip()
        revision = parameters.get("revision", ["main"])[0].strip() or "main"
        if not _valid_model_id(model_id):
            self._send_error(400, "Enter a valid Hugging Face model ID")
            return
        if not REVISION_PATTERN.fullmatch(revision):
            self._send_error(400, "Enter a valid revision")
            return
        request_token, present, valid, from_cookie = self._user_token()
        if not valid:
            message = "The saved Hugging Face session was invalid" if from_cookie else "Use a valid Hugging Face Bearer token"
            self._send_error(400, message, expire_cookie=from_cookie)
            return
        token = request_token if present else self.server.state.server_token
        cache_allowed = not present and not self.server.state.server_token
        cache_key = f"{model_id}@{revision}"
        if cache_allowed:
            cached = self.server.state.cache_get(cache_key)
            if cached is not None:
                self._send(200, "application/json", cached, no_store=True)
                return
        try:
            result = inspect_model(model_id, revision, token)
        except InspectionError as error:
            expire = from_cookie and error.status in {401, 403}
            if expire:
                self._discard_request_session()
            self._send_error(_api_error_status(error.status), str(error), expire_cookie=expire)
            return
        except Exception as error:  # Keep the HTTP worker alive on malformed upstream data.
            self._send_error(502, f"Model inspection failed: {error}")
            return
        body = _json_bytes(result)
        if cache_allowed:
            self.server.state.cache_put(cache_key, body)
        self._send(200, "application/json", body, no_store=True)

    def _handle_debug_case_list(self) -> None:
        self._send_json(200, {"cases": self.server.state.debug_store.list_cases()})

    def _handle_debug_dataset(self, query: str) -> None:
        parameters = parse_qs(query, keep_blank_values=True)
        dataset = parameters.get("dataset", [""])[0].strip()
        requested_config = parameters.get("config", [""])[0].strip()
        requested_split = parameters.get("split", [""])[0].strip()
        if not _valid_model_id(dataset):
            self._send_error(400, "Enter a valid Hugging Face dataset ID")
            return
        token, _present, valid, from_cookie = self._user_token()
        if not valid:
            self._send_error(400, "The saved Hugging Face session was invalid", expire_cookie=from_cookie)
            return
        selection = _dataset_selection(dataset, requested_config, requested_split, token)
        if isinstance(selection, str):
            self._send_error(502, selection)
            return
        config, split = selection
        url = "https://datasets-server.huggingface.co/rows?" + urlencode({
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": 0,
            "length": 100,
        })
        response = http_request(url, method="GET", token=token, timeout=60)
        payload = _runtime_payload(response)
        if response.error:
            self._send_error(502, f"Could not reach the Hugging Face dataset viewer: {response.error}")
            return
        if response.status != 200 or payload is None:
            message = payload.get("error") if isinstance(payload, dict) else None
            self._send_error(_api_error_status(response.status), message or f"Dataset viewer returned {response.status}")
            return
        self._send_json(200, {
            "dataset": dataset,
            "config": config,
            "split": split,
            "features": payload.get("features", []),
            "rows": [item.get("row", {}) for item in payload.get("rows", []) if isinstance(item, dict)],
            "totalRows": payload.get("num_rows_total"),
            "partial": payload.get("partial", False),
        })

    def _handle_debug_case_get(self, case_id: str) -> None:
        record = self.server.state.debug_store.get_case(case_id)
        if record is None:
            self._send_error(404, "Debug case not found")
            return
        self._send_json(200, record)

    def _handle_debug_case_create(self) -> None:
        payload = self._request_json()
        if payload is None:
            return
        try:
            record = self.server.state.debug_store.create_case(payload)
        except DebugStoreError as error:
            self._send_error(400, str(error))
            return
        self._send_json(201, record)

    def _handle_debug_case_replace(self, case_id: str) -> None:
        payload = self._request_json()
        if payload is None:
            return
        try:
            record = self.server.state.debug_store.replace_case(case_id, payload)
        except DebugStoreError as error:
            self._send_error(400, str(error))
            return
        if record is None:
            self._send_error(404, "Debug case not found")
            return
        self._send_json(200, record)

    def _handle_debug_case_delete(self, case_id: str) -> None:
        if not self.server.state.debug_store.delete_case(case_id):
            self._send_error(404, "Debug case not found")
            return
        self._send_json(200, {"ok": True, "id": case_id})

    def _request_json(self) -> dict | None:
        value = self.headers.get("Content-Length", "")
        try:
            length = int(value)
        except ValueError:
            self._send_error(400, "A valid JSON request body is required")
            return None
        if length <= 0 or length > MAX_JSON_REQUEST_BYTES:
            self._send_error(413 if length > MAX_JSON_REQUEST_BYTES else 400, "JSON request body is missing or too large")
            return None
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error(400, "Request body must be valid JSON")
            return None
        if not isinstance(payload, dict):
            self._send_error(400, "Request body must be a JSON object")
            return None
        return payload

    def _runtime_session(self) -> RuntimeSession | None:
        identifier = self._cookie_value(RUNTIME_COOKIE_NAME)
        if not identifier or not TOKEN_PATTERN.fullmatch(identifier):
            return None
        return self.server.state.runtime_session(identifier)

    def _handle_runtime_connect(self) -> None:
        payload = self._request_json()
        if payload is None:
            return
        endpoint = _runtime_endpoint(str(payload.get("endpoint", "")))
        secret = str(payload.get("secret", "")).strip()
        if not endpoint:
            self._send_error(400, "Use the loopback URL shown for the local execution worker")
            return
        if not secret and endpoint.startswith("http://"):
            secret = self.server.state.local_worker_secret(endpoint)
        if not TOKEN_PATTERN.fullmatch(secret):
            self._send_error(400, "Start the local worker with the command shown in Settings")
            return
        response = runtime_request(endpoint, "/health", secret, method="GET", timeout=30)
        worker = _runtime_payload(response)
        if response.error:
            self._send_error(502, f"Could not reach the execution worker: {response.error}")
            return
        if response.status != 200 or worker is None:
            message = worker.get("error") if isinstance(worker, dict) else None
            self._send_error(502, message or f"Execution worker health check failed ({response.status})")
            return
        if not _worker_version_supported(worker):
            self._send_error(409, _worker_upgrade_message(worker))
            return
        previous = self._cookie_value(RUNTIME_COOKIE_NAME)
        if previous:
            self.server.state.remove_runtime_session(previous, cleanup=True)
        identifier = self.server.state.create_runtime_session(endpoint, secret)
        self._send_json(
            200,
            {"connected": True, "reachable": True, "endpoint": endpoint, "worker": worker},
            set_cookie=_runtime_cookie(identifier),
        )

    def _handle_daytona_recommend(self) -> None:
        payload = self._request_json()
        if payload is not None:
            self._send_json(200, recommend_gpu(payload))

    def _handle_daytona_provision(self) -> None:
        payload = self._request_json()
        if payload is None:
            return
        raw_api_key = payload.get("apiKey", "")
        api_key = raw_api_key.strip() if isinstance(raw_api_key, str) else ""
        try:
            runtime = provision_runtime(
                api_key,
                payload,
                worker_path=PROJECT_ROOT / "workers" / "modeldebugger_worker.py",
            )
        except DaytonaError as error:
            self._send_error(502, str(error))
            return
        previous = self._cookie_value(RUNTIME_COOKIE_NAME)
        if previous:
            self.server.state.remove_runtime_session(previous, cleanup=True)
        identifier = self.server.state.create_runtime_session(
            runtime.endpoint,
            runtime.secret,
            provider="daytona",
            preview_token=runtime.preview_token,
            sandbox_id=runtime.sandbox_id,
            provider_key=api_key or os.environ.get("DAYTONA_API_KEY", "").strip(),
            gpu_type=runtime.gpu_type,
            quantization=str(runtime.recommendation.get("quantization", "none")),
        )
        self._send_json(
            200,
            {
                "connected": True,
                "reachable": True,
                "provider": "daytona",
                "sandboxId": runtime.sandbox_id,
                "gpuType": runtime.gpu_type,
                "quantization": runtime.recommendation.get("quantization", "none"),
                "recommendation": runtime.recommendation,
                "worker": {"ok": True, "accelerator": runtime.gpu_type, "modelLoaded": False},
            },
            set_cookie=_runtime_cookie(identifier),
        )

    def _handle_runtime_status(self) -> None:
        session = self._runtime_session()
        if session is None:
            self._send_json(200, {"connected": False, "reachable": False})
            return
        response = runtime_request(
            session.endpoint,
            "/health",
            session.secret,
            method="GET",
            timeout=20,
            preview_token=session.preview_token,
        )
        worker = _runtime_payload(response)
        if response.error or response.status != 200 or worker is None:
            message = response.error or (worker.get("error") if isinstance(worker, dict) else None)
            self._send_json(
                200,
                {
                    "connected": True,
                    "reachable": False,
                    "provider": session.provider,
                    "gpuType": session.gpu_type or None,
                    "quantization": session.quantization,
                    "error": message or f"Worker returned {response.status}",
                },
            )
            return
        if not _worker_version_supported(worker):
            self._send_json(
                200,
                {
                    "connected": True,
                    "reachable": False,
                    "provider": session.provider,
                    "gpuType": session.gpu_type or None,
                    "quantization": session.quantization,
                    "worker": worker,
                    "error": _worker_upgrade_message(worker),
                },
            )
            return
        self._send_json(
            200,
            {
                "connected": True,
                "reachable": True,
                "endpoint": session.endpoint if session.provider == "local" else None,
                "provider": session.provider,
                "gpuType": session.gpu_type or None,
                "quantization": session.quantization,
                "worker": worker,
            },
        )

    def _handle_runtime_disconnect(self) -> None:
        identifier = self._cookie_value(RUNTIME_COOKIE_NAME)
        warning = ""
        if identifier:
            runtime = self.server.state.remove_runtime_session(identifier)
            if runtime is not None:
                warning = self.server.state._cleanup_runtime(runtime)
        self._send_json(
            200,
            {"ok": True, "connected": False, "warning": warning or None},
            set_cookie=_expired_runtime_cookie(),
        )

    def _handle_runtime_proxy(self, worker_path: str) -> None:
        session = self._runtime_session()
        if session is None:
            self._send_error(401, "Connect an execution worker first", expire_runtime_cookie=True)
            return
        payload = self._request_json()
        if payload is None:
            return
        response = runtime_request(
            session.endpoint,
            worker_path,
            session.secret,
            method="POST",
            payload=payload,
            timeout=RUNTIME_TIMEOUT_SECONDS,
            preview_token=session.preview_token,
        )
        if response.error:
            self._send_error(502, f"The execution worker is unavailable: {response.error}")
            return
        content_type = response.header("Content-Type", "application/json")
        self._send(response.status or 502, content_type, response.body, no_store=True)

    def _discard_request_session(self) -> None:
        session_id = self._cookie_value(COOKIE_NAME)
        if session_id:
            self.server.state.remove_session(session_id)

    def _serve_static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        item = STATIC_FILES.get(path)
        if item is None:
            self._send_error(404, "Not found", no_store=False)
            return
        relative_path, content_type = item
        try:
            body = (PROJECT_ROOT / relative_path).read_bytes()
        except OSError:
            self._send_error(404, "Not found", no_store=False)
            return
        self._send(200, content_type, body, no_store=False)

    def _send_error(
        self,
        status: int,
        message: str,
        *,
        expire_cookie: bool = False,
        expire_runtime_cookie: bool = False,
        no_store: bool = True,
    ) -> None:
        self._send_json(
            status,
            {"error": message},
            set_cookie=_expired_cookie() if expire_cookie else _expired_runtime_cookie() if expire_runtime_cookie else None,
            no_store=no_store,
        )

    def _send_json(self, status: int, payload: object, *, set_cookie: str | None = None, no_store: bool = True) -> None:
        self._send(status, "application/json", _json_bytes(payload), no_store=no_store, set_cookie=set_cookie)

    def _send(
        self,
        status: int,
        content_type: str,
        body: bytes,
        *,
        no_store: bool,
        set_cookie: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if no_store:
            self.send_header("Cache-Control", "no-store")
        if set_cookie:
            self.send_header("Set-Cookie", set_cookie)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self.close_connection = True


def _valid_model_id(model_id: str) -> bool:
    if not MODEL_PATTERN.fullmatch(model_id):
        return False
    return all(segment not in {".", ".."} for segment in model_id.split("/"))


def _debug_case_id(path: str) -> str | None:
    match = re.fullmatch(r"/api/debug/cases/([a-f0-9]{32})", path)
    return match.group(1) if match else None


def _api_error_status(upstream_status: int) -> int:
    return upstream_status if upstream_status in {401, 403, 404} else 502


def _dataset_selection(dataset: str, config: str, split: str, token: str) -> tuple[str, str] | str:
    """Resolve an omitted dataset subset/split through the official dataset-viewer API."""
    if config and split:
        return config, split
    response = http_request(
        "https://datasets-server.huggingface.co/splits?" + urlencode({"dataset": dataset}),
        method="GET",
        token=token,
        timeout=60,
    )
    payload = _runtime_payload(response)
    if response.error:
        return f"Could not reach the Hugging Face dataset viewer: {response.error}"
    if response.status != 200 or payload is None:
        message = payload.get("error") if isinstance(payload, dict) else None
        return str(message or f"Dataset viewer returned {response.status}")
    choices = [item for item in payload.get("splits", []) if isinstance(item, dict)]
    if config:
        choices = [item for item in choices if str(item.get("config", "")) == config]
    if split:
        choices = [item for item in choices if str(item.get("split", "")) == split]
    if not choices:
        return "The dataset viewer did not expose a matching subset and split"
    selected = choices[0]
    return str(selected.get("config", "default")), str(selected.get("split", "train"))


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _runtime_endpoint(value: str) -> str:
    """Accept only an explicit loopback worker to prevent SSRF."""
    parsed = urlsplit(value.strip())
    hostname = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError:
        return ""
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        return ""
    if parsed.scheme == "http" and hostname in {"127.0.0.1", "localhost", "::1"} and port is not None and 1024 <= port <= 65535:
        host = f"[{hostname}]" if ":" in hostname else hostname
        return f"http://{host}:{port}"
    return ""


def runtime_request(
    endpoint: str,
    path: str,
    secret: str,
    *,
    method: str,
    payload: dict | None = None,
    timeout: float,
    preview_token: str = "",
) -> HTTPResponse:
    body = _json_bytes(payload) if payload is not None else None
    return http_request(
        f"{endpoint}{path}",
        method=method,
        token=secret,
        headers={"x-daytona-preview-token": preview_token} if preview_token else None,
        body=body,
        timeout=timeout,
    )


def _runtime_payload(response: HTTPResponse) -> dict | None:
    try:
        payload = json.loads(response.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _worker_version_supported(worker: dict) -> bool:
    """Reject known-old bundled workers while allowing unversioned integrations."""
    version = str(worker.get("version", "")).strip()
    if not version:
        return True
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", version)
    if match is None:
        return False
    return tuple(int(part) for part in match.groups()) >= MINIMUM_WORKER_VERSION


def _worker_upgrade_message(worker: dict) -> str:
    version = str(worker.get("version", "unknown")).strip() or "unknown"
    minimum = ".".join(str(part) for part in MINIMUM_WORKER_VERSION)
    return (
        f"Worker {version} is outdated; ModelDebugger requires {minimum} or newer. "
        "Stop the existing worker and run `make worker` again."
    )


def _persistent_cookie(session_id: str) -> str:
    return (
        f"{COOKIE_NAME}={session_id}; Path={COOKIE_PATH}; Max-Age={SESSION_SECONDS}; "
        "HttpOnly; SameSite=Strict"
    )


def _expired_cookie() -> str:
    return (
        f"{COOKIE_NAME}=; Path={COOKIE_PATH}; Max-Age=0; "
        "Expires=Thu, 01 Jan 1970 00:00:00 GMT; HttpOnly; SameSite=Strict"
    )


def _runtime_cookie(session_id: str) -> str:
    return (
        f"{RUNTIME_COOKIE_NAME}={session_id}; Path={RUNTIME_COOKIE_PATH}; "
        f"Max-Age={RUNTIME_SESSION_SECONDS}; HttpOnly; SameSite=Strict"
    )


def _expired_runtime_cookie() -> str:
    return (
        f"{RUNTIME_COOKIE_NAME}=; Path={RUNTIME_COOKIE_PATH}; Max-Age=0; "
        "Expires=Thu, 01 Jan 1970 00:00:00 GMT; HttpOnly; SameSite=Strict"
    )


def configured_port() -> int:
    try:
        port = int(os.environ.get("PORT", DEFAULT_PORT))
    except ValueError:
        return DEFAULT_PORT
    return port if 1 <= port <= 65535 else DEFAULT_PORT


def run_server() -> None:
    port = configured_port()
    data_directory = Path(os.environ.get("MODELDEBUGGER_DATA_DIR", PROJECT_ROOT / ".modeldebugger"))
    state = ApplicationState(
        os.environ.get("HF_TOKEN", "").strip(),
        data_directory / "modeldebugger.sqlite3",
    )
    server = RefusalScopeServer(("127.0.0.1", port), state)
    print(f"ModelDebugger is running at http://localhost:{port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.close()
