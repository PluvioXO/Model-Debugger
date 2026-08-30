from __future__ import annotations

import json
import tempfile
import threading
import unittest
from dataclasses import dataclass
from email.message import Message
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener
from unittest.mock import patch

from refusalscope.daytona import ProvisionedRuntime
from refusalscope.http_client import HTTPResponse
from refusalscope.server import (
    ApplicationState,
    RefusalScopeServer,
    _runtime_endpoint,
    _worker_version_supported,
)


@dataclass(slots=True)
class Response:
    status: int
    headers: Message
    body: bytes


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.account_patcher = patch(
            "refusalscope.server.inspect_account",
            return_value={
                "name": "tester",
                "fullname": "Test User",
                "avatarUrl": "https://example.test/avatar.png",
                "type": "user",
                "isPro": False,
                "orgs": [],
            },
        )
        cls.model_patcher = patch(
            "refusalscope.server.inspect_model",
            return_value={"graph": {"name": "test/model", "nodes": [], "edges": []}},
        )
        cls.inspect_account = cls.account_patcher.start()
        cls.inspect_model = cls.model_patcher.start()
        cls.server = RefusalScopeServer(("127.0.0.1", 0), ApplicationState())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.model_patcher.stop()
        cls.account_patcher.stop()

    def setUp(self) -> None:
        self.cookies = CookieJar()
        self.opener = build_opener(HTTPCookieProcessor(self.cookies))
        self.inspect_model.reset_mock()

    def request(self, path: str, *, method: str = "GET", headers: dict[str, str] | None = None, json_body: dict | None = None):
        request_headers = dict(headers or {})
        data = json.dumps(json_body).encode() if json_body is not None else b"" if method == "POST" else None
        if json_body is not None:
            request_headers["Content-Type"] = "application/json"
        request = Request(
            self.base_url + path,
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with self.opener.open(request, timeout=5) as response:
                return Response(response.status, response.headers, response.read())
        except HTTPError as error:
            with error:
                return Response(error.status, error.headers, error.read())

    def test_health_and_static_allowlist(self) -> None:
        health = self.request("/api/health")
        self.assertEqual(health.status, 200)
        self.assertEqual(json.loads(health.body), {"backend": "python", "ok": True})
        self.assertEqual(health.headers["Cache-Control"], "no-store")
        self.assertIn("ModelDebugger", self.request("/").body.decode())
        self.assertIn("routeGraphEdge", self.request("/src/graph-routing.js").body.decode())
        self.assertIn("normaliseBenchmarkExamples", self.request("/src/benchmark.js").body.decode())
        diagnostic = self.request("/demo/gpt2-capital-diagnostic")
        self.assertEqual(diagnostic.status, 200)
        self.assertIn("Where does GPT", diagnostic.body.decode())
        self.assertIn("GPT2_DIAGNOSTIC", self.request("/src/gpt2-diagnostic-data.js").body.decode())
        self.assertEqual(self.request("/gpt2-diagnostic.css").headers["Content-Type"], "text/css; charset=utf-8")
        tutorial_capture = self.request("/assets/tutorial/03-paired-comparison.png")
        self.assertEqual(tutorial_capture.status, 200)
        self.assertEqual(tutorial_capture.headers["Content-Type"], "image/png")
        self.assertTrue(tutorial_capture.body.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(self.request("/colab/ModelDebugger_Worker.ipynb").status, 404)
        self.assertEqual(self.request("/colab/modeldebugger_worker.py").status, 404)
        self.assertEqual(self.request("/src/model.js").status, 404)

    def test_authentication_cookie_profile_and_logout(self) -> None:
        self.assertEqual(self.request("/api/huggingface/account").status, 401)
        connected = self.request(
            "/api/huggingface/account", headers={"Authorization": "Bearer hf_test-token"}
        )
        self.assertEqual(connected.status, 200)
        self.assertEqual(json.loads(connected.body)["avatarUrl"], "https://example.test/avatar.png")
        set_cookie = connected.headers["Set-Cookie"]
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("SameSite=Strict", set_cookie)
        self.assertIn("Max-Age=2592000", set_cookie)
        account_cookies = connected.headers.get_all("Set-Cookie")
        self.assertEqual(len(account_cookies), 2)
        self.assertTrue(any("refusalscope_hf_daytona_session=" in cookie for cookie in account_cookies))
        self.assertTrue(any("Path=/api/runtime/daytona/provision" in cookie for cookie in account_cookies))

        restored = self.request("/api/huggingface/account")
        self.assertEqual(restored.status, 200)
        self.assertEqual(json.loads(restored.body)["name"], "tester")

        logged_out = self.request("/api/huggingface/logout", method="POST")
        self.assertEqual(logged_out.status, 200)
        logout_cookies = logged_out.headers.get_all("Set-Cookie")
        self.assertEqual(len(logout_cookies), 2)
        self.assertTrue(all("Max-Age=0" in cookie for cookie in logout_cookies))
        self.assertEqual(self.request("/api/huggingface/account").status, 401)

    def test_request_validation(self) -> None:
        self.assertEqual(
            self.request("/api/huggingface/account", headers={"Authorization": "Basic invalid"}).status,
            400,
        )
        self.assertEqual(
            self.request(
                "/api/huggingface/account", headers={"Cookie": "refusalscope_hf_session=invalid%session"}
            ).status,
            400,
        )
        self.assertEqual(self.request("/api/huggingface/logout").status, 405)
        self.assertEqual(self.request("/api/huggingface?model=..%2Fmodel&revision=main").status, 400)
        self.assertEqual(self.request("/api/huggingface?model=org/model&revision=bad%20revision").status, 400)

    def test_debug_case_crud(self) -> None:
        created_response = self.request(
            "/api/debug/cases",
            method="POST",
            json_body={
                "name": "Incorrect capital",
                "model": {"modelId": "test/model", "revision": "main"},
                "failure": {"prompt": "The capital of France is"},
                "expected": {"text": " Paris"},
            },
        )
        self.assertEqual(created_response.status, 201)
        created = json.loads(created_response.body)
        case_id = created["id"]

        listed = json.loads(self.request("/api/debug/cases").body)["cases"]
        self.assertTrue(any(item["id"] == case_id for item in listed))
        self.assertEqual(
            json.loads(self.request(f"/api/debug/cases/{case_id}").body)["expected"]["text"],
            " Paris",
        )

        updated_response = self.request(
            f"/api/debug/cases/{case_id}",
            method="PUT",
            json_body={**created, "name": "Updated capital case", "status": "investigating"},
        )
        self.assertEqual(updated_response.status, 200)
        self.assertEqual(json.loads(updated_response.body)["name"], "Updated capital case")

        deleted = self.request(f"/api/debug/cases/{case_id}", method="DELETE")
        self.assertEqual(deleted.status, 200)
        self.assertEqual(self.request(f"/api/debug/cases/{case_id}").status, 404)

    @patch("refusalscope.server.http_request")
    def test_hugging_face_dataset_import_resolves_default_split(self, request_upstream) -> None:
        request_upstream.side_effect = [
            HTTPResponse(
                status=200,
                body=json.dumps({"splits": [{"config": "default", "split": "train"}]}).encode(),
            ),
            HTTPResponse(
                status=200,
                body=json.dumps({
                    "features": [{"name": "prompt"}],
                    "rows": [{"row_idx": 0, "row": {"prompt": "Test", "expected": "Answer"}}],
                    "num_rows_total": 1,
                }).encode(),
            ),
        ]
        response = self.request("/api/huggingface/dataset?dataset=test%2Ffailures")
        self.assertEqual(response.status, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["config"], "default")
        self.assertEqual(payload["split"], "train")
        self.assertEqual(payload["rows"][0]["prompt"], "Test")

    def test_public_models_are_cached_and_authenticated_models_are_not(self) -> None:
        path = "/api/huggingface?model=test/model&revision=main"
        self.assertEqual(self.request(path).status, 200)
        self.assertEqual(self.request(path).status, 200)
        self.assertEqual(self.inspect_model.call_count, 1)

        self.request("/api/huggingface/account", headers={"Authorization": "Bearer hf_test-token"})
        self.assertEqual(self.request(path).status, 200)
        self.assertEqual(self.request(path).status, 200)
        self.assertEqual(self.inspect_model.call_count, 3)
        self.assertEqual(self.inspect_model.call_args.args[2], "hf_test-token")

    def test_runtime_endpoint_is_restricted_to_loopback(self) -> None:
        self.assertEqual(_runtime_endpoint("https://worker-name.trycloudflare.com"), "")
        self.assertEqual(_runtime_endpoint("http://127.0.0.1:8765"), "http://127.0.0.1:8765")
        self.assertEqual(_runtime_endpoint("http://localhost:9000/"), "http://localhost:9000")
        self.assertEqual(_runtime_endpoint("http://[::1]:8765"), "http://[::1]:8765")
        self.assertEqual(_runtime_endpoint("http://worker-name.trycloudflare.com"), "")
        self.assertEqual(_runtime_endpoint("https://example.com"), "")
        self.assertEqual(_runtime_endpoint("http://192.168.1.8:8765"), "")
        self.assertEqual(_runtime_endpoint("http://localhost:80"), "")
        self.assertEqual(_runtime_endpoint("https://worker-name.trycloudflare.com/path"), "")

    def test_known_stale_worker_versions_are_rejected(self) -> None:
        self.assertFalse(_worker_version_supported({"version": "0.3.0"}))
        self.assertFalse(_worker_version_supported({"version": "0.3.1"}))
        self.assertTrue(_worker_version_supported({"version": "0.4.0-beta.1"}))
        self.assertFalse(_worker_version_supported({"version": "development"}))
        self.assertTrue(_worker_version_supported({}))

    @patch("refusalscope.server.runtime_request")
    def test_runtime_connection_proxy_and_disconnect(self, request_worker) -> None:
        request_worker.return_value = HTTPResponse(
            status=200,
            body=json.dumps({"ok": True, "accelerator": "Test GPU", "modelLoaded": False}).encode(),
        )
        connected = self.request(
            "/api/runtime/connect",
            method="POST",
            json_body={"endpoint": "http://127.0.0.1:8765", "secret": "worker-secret"},
        )
        self.assertEqual(connected.status, 200)
        self.assertTrue(json.loads(connected.body)["reachable"])
        self.assertIn("HttpOnly", connected.headers["Set-Cookie"])

        status = self.request("/api/runtime/status")
        self.assertEqual(status.status, 200)
        self.assertTrue(json.loads(status.body)["connected"])

        request_worker.return_value = HTTPResponse(status=200, body=b'{"ok":true,"runId":"test-run"}')
        forwarded = self.request(
            "/api/runtime/forward", method="POST", json_body={"prompt": "Test prompt"}
        )
        self.assertEqual(forwarded.status, 200)
        self.assertEqual(json.loads(forwarded.body)["runId"], "test-run")

        for route in ("generate", "sweep"):
            request_worker.return_value = HTTPResponse(status=200, body=json.dumps({"ok": True, "route": route}).encode())
            proxied = self.request(f"/api/runtime/{route}", method="POST", json_body={"test": True})
            self.assertEqual(proxied.status, 200)
            self.assertEqual(json.loads(proxied.body)["route"], route)

        disconnected = self.request("/api/runtime/disconnect", method="POST", json_body={})
        self.assertEqual(disconnected.status, 200)
        self.assertFalse(json.loads(disconnected.body)["connected"])

    @patch("refusalscope.server.delete_runtime")
    @patch("refusalscope.server.runtime_request")
    @patch("refusalscope.server.provision_runtime")
    def test_daytona_key_is_server_only_and_disconnect_deletes_sandbox(
        self,
        provision,
        request_worker,
        delete,
    ) -> None:
        provision.return_value = ProvisionedRuntime(
            endpoint="https://8765-test.proxy.daytona.works",
            preview_token="preview-secret",
            secret="worker-secret",
            sandbox_id="sandbox-test",
            gpu_type="RTX-4090",
            recommendation={"recommendedGpu": "RTX-4090", "quantization": "none"},
        )
        provisioned = self.request(
            "/api/runtime/daytona/provision",
            method="POST",
            json_body={
                "apiKey": "daytona-user-key",
                "gpuType": "auto",
                "modelId": "test/model",
                "parameterCount": 1_000_000_000,
                "checkpointBytes": 2_000_000_000,
                "hfToken": "browser-injected-token",
            },
        )
        self.assertEqual(provisioned.status, 200)
        payload = json.loads(provisioned.body)
        self.assertEqual(payload["provider"], "daytona")
        self.assertNotIn("apiKey", payload)
        self.assertNotIn("preview-secret", provisioned.body.decode())
        self.assertNotIn("worker-secret", provisioned.body.decode())
        self.assertNotIn("hfToken", provision.call_args.args[1])

        request_worker.return_value = HTTPResponse(status=200, body=b'{"ok":true,"runId":"daytona-run"}')
        forwarded = self.request("/api/runtime/forward", method="POST", json_body={"prompt": "Test"})
        self.assertEqual(forwarded.status, 200)
        self.assertEqual(request_worker.call_args.kwargs["preview_token"], "preview-secret")

        disconnected = self.request("/api/runtime/disconnect", method="POST", json_body={})
        self.assertEqual(disconnected.status, 200)
        delete.assert_called_once_with("daytona-user-key", "sandbox-test")

    @patch("refusalscope.server.delete_runtime")
    @patch("refusalscope.server.provision_runtime")
    def test_daytona_inherits_the_validated_hugging_face_session(self, provision, delete) -> None:
        provision.return_value = ProvisionedRuntime(
            endpoint="https://8765-test.proxy.daytona.works",
            preview_token="preview-secret",
            secret="worker-secret",
            sandbox_id="sandbox-inherited-token",
            gpu_type="RTX-4090",
            recommendation={"recommendedGpu": "RTX-4090", "quantization": "none"},
        )
        connected = self.request(
            "/api/huggingface/account", headers={"Authorization": "Bearer hf_inherited-token"}
        )
        self.assertEqual(connected.status, 200)

        provisioned = self.request(
            "/api/runtime/daytona/provision",
            method="POST",
            json_body={
                "apiKey": "daytona-user-key",
                "gpuType": "auto",
                "modelId": "private/model",
                "parameterCount": 1_000_000_000,
                "checkpointBytes": 2_000_000_000,
            },
        )
        self.assertEqual(provisioned.status, 200)
        self.assertEqual(provision.call_args.args[1]["hfToken"], "hf_inherited-token")
        response_payload = json.loads(provisioned.body)
        self.assertEqual(response_payload["huggingFaceAccess"], "connected-account")
        self.assertNotIn("hf_inherited-token", provisioned.body.decode())

        disconnected = self.request("/api/runtime/disconnect", method="POST", json_body={})
        self.assertEqual(disconnected.status, 200)
        delete.assert_called_once_with("daytona-user-key", "sandbox-inherited-token")

    def test_daytona_recommendation_route_does_not_provision(self) -> None:
        response = self.request(
            "/api/runtime/daytona/recommend",
            method="POST",
            json_body={"parameterCount": 1_000_000_000, "checkpointBytes": 2_000_000_000},
        )
        self.assertEqual(response.status, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["recommendedGpu"], "RTX-4090")
        self.assertEqual(payload["quantization"], "none")

    @patch("refusalscope.server.provision_runtime")
    @patch("refusalscope.server.validate_api_key")
    def test_daytona_api_key_check_never_provisions_or_returns_the_key(self, validate, provision) -> None:
        validate.return_value = {
            "valid": True,
            "message": "Daytona accepted this API key. No sandbox was created.",
        }
        response = self.request(
            "/api/runtime/daytona/validate",
            method="POST",
            json_body={"apiKey": "daytona-secret-key"},
        )
        self.assertEqual(response.status, 200)
        payload = json.loads(response.body)
        self.assertTrue(payload["valid"])
        self.assertNotIn("daytona-secret-key", response.body.decode())
        validate.assert_called_once_with("daytona-secret-key")
        provision.assert_not_called()

    @patch("refusalscope.server.runtime_request")
    def test_local_worker_can_be_discovered_without_pasting_a_secret(self, request_worker) -> None:
        request_worker.return_value = HTTPResponse(
            status=200,
            body=json.dumps({"ok": True, "version": "0.4.0", "modelLoaded": False}).encode(),
        )
        with tempfile.TemporaryDirectory() as directory:
            session_file = Path(directory) / "local-worker.json"
            session_file.write_text(json.dumps({
                "endpoint": "http://127.0.0.1:8765",
                "secret": "local-test-secret",
            }))
            previous = self.server.state.local_worker_session
            self.server.state.local_worker_session = session_file
            try:
                connected = self.request(
                    "/api/runtime/connect",
                    method="POST",
                    json_body={"endpoint": "http://127.0.0.1:8765", "secret": ""},
                )
            finally:
                self.server.state.local_worker_session = previous
        self.assertEqual(connected.status, 200)
        self.assertEqual(request_worker.call_args.args[2], "local-test-secret")

    @patch("refusalscope.server.runtime_request")
    def test_stale_worker_connection_explains_how_to_restart(self, request_worker) -> None:
        request_worker.return_value = HTTPResponse(
            status=200,
            body=json.dumps({"ok": True, "version": "0.3.0", "modelLoaded": True}).encode(),
        )
        connected = self.request(
            "/api/runtime/connect",
            method="POST",
            json_body={"endpoint": "http://127.0.0.1:8765", "secret": "worker-secret"},
        )
        self.assertEqual(connected.status, 409)
        self.assertIn("make worker", json.loads(connected.body)["error"])


if __name__ == "__main__":
    unittest.main()
