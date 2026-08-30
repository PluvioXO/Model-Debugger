from __future__ import annotations

import unittest

from refusalscope.daytona import DAYTONA_GPU_ORDER, _sandbox_create_params, recommend_gpu


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


if __name__ == "__main__":
    unittest.main()
