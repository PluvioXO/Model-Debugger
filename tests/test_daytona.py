from __future__ import annotations

import unittest
from unittest.mock import patch

from refusalscope.daytona import DAYTONA_GPU_ORDER, _provisioning_error, _sandbox_create_params, recommend_gpu, validate_api_key


class DaytonaRecommendationTests(unittest.TestCase):
    def test_provisioning_profile_is_always_spot(self) -> None:
        class Value:
            def __init__(self, value):
                self.value = value

        class Record:
            def __init__(self, **values):
                self.values = values

        params = _sandbox_create_params(
            Record,
            Record,
            Value,
            ["RTX-4090"],
            {"MODELDEBUGGER_WORKER_SECRET": "secret"},
        )
        self.assertIs(params.values["spot"], True)
        self.assertIs(params.values["public"], False)
        self.assertIs(params.values["ephemeral"], True)
        self.assertEqual(params.values["auto_stop_interval"], 0)
        self.assertNotIn("auto_delete_interval", params.values)
        self.assertEqual(params.values["resources"].values["gpu"], 1)

    def test_missing_spot_credits_never_suggests_an_on_demand_fallback(self) -> None:
        error = _provisioning_error(RuntimeError("Organization doesn't have spot GPU credits. Add more."))
        message = str(error)
        self.assertIn("does not have spot GPU credits", message)
        self.assertIn("will not fall back to on-demand", message)
        self.assertIn("Local machine", message)

    def test_small_model_uses_the_smallest_supported_gpu(self) -> None:
        result = recommend_gpu({"parameterCount": 1_000_000_000, "checkpointBytes": 2_000_000_000})
        self.assertEqual(result["recommendedGpu"], "RTX-4090")
        self.assertEqual(result["quantization"], "none")
        self.assertTrue(result["fitsSingleGpu"])

    def test_large_model_switches_to_4bit_before_claiming_it_fits(self) -> None:
        result = recommend_gpu({"parameterCount": 70_000_000_000, "checkpointBytes": 140_000_000_000})
        self.assertEqual(result["recommendedGpu"], "H100")
        self.assertEqual(result["quantization"], "4bit")
        self.assertTrue(result["fitsSingleGpu"])

    def test_oversized_model_fails_closed(self) -> None:
        result = recommend_gpu({"parameterCount": 400_000_000_000, "checkpointBytes": 800_000_000_000})
        self.assertEqual(result["recommendedGpu"], "H200")
        self.assertFalse(result["fitsSingleGpu"])
        self.assertIn("smaller checkpoint", result["reason"])

    def test_empty_checkpoint_is_a_baseline_not_a_false_measurement(self) -> None:
        result = recommend_gpu({})
        self.assertEqual(result["recommendedGpu"], DAYTONA_GPU_ORDER[0])
        self.assertEqual(result["confidence"], "baseline")
        self.assertIsNone(result["estimatedPeakBytes"])

    @patch("refusalscope.daytona._daytona_sdk")
    def test_api_key_check_uses_one_read_only_list_request(self, sdk) -> None:
        calls = {}

        class Config:
            def __init__(self, *, api_key):
                calls["api_key"] = api_key

        class Query:
            def __init__(self, *, limit):
                calls["limit"] = limit

        class AuthenticationError(Exception):
            pass

        class ForbiddenError(Exception):
            pass

        class Client:
            def __init__(self, _config):
                pass

            def list(self, query, request_timeout=None):
                calls["query"] = query
                calls["request_timeout"] = request_timeout
                return iter([])

        sdk.return_value = (Client, Config, None, None, None, None, Query, AuthenticationError, ForbiddenError)
        result = validate_api_key("  daytona-test-key  ")
        self.assertTrue(result["valid"])
        self.assertIn("No sandbox was created", result["message"])
        self.assertEqual(calls["api_key"], "daytona-test-key")
        self.assertEqual(calls["limit"], 1)
        self.assertEqual(calls["request_timeout"], 20)

    @patch("refusalscope.daytona._daytona_sdk")
    def test_api_key_check_distinguishes_rejection_from_missing_permissions(self, sdk) -> None:
        class Config:
            def __init__(self, *, api_key):
                self.api_key = api_key

        class Query:
            def __init__(self, *, limit):
                self.limit = limit

        class AuthenticationError(Exception):
            pass

        class ForbiddenError(Exception):
            pass

        class Client:
            error = AuthenticationError("unauthorized")

            def __init__(self, _config):
                pass

            def list(self, _query, request_timeout=None):
                raise self.error

        sdk.return_value = (Client, Config, None, None, None, None, Query, AuthenticationError, ForbiddenError)
        self.assertIn("rejected", validate_api_key("bad-key")["message"])
        Client.error = ForbiddenError("forbidden")
        self.assertIn("cannot access sandboxes", validate_api_key("limited-key")["message"])


if __name__ == "__main__":
    unittest.main()
