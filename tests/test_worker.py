from __future__ import annotations

import importlib.util
import unittest

from workers.modeldebugger_worker import (
    _attribution_summary,
    _layer_index,
    _logit_lens_timeline,
    _metric_value_identity,
    _module_category,
    _normalise_metric,
    _normalise_generation_settings,
    _normalise_lens_settings,
    _normalise_sweep_axes,
    _preflight_model_size,
    _register_normalization_trace,
    _sample_stage_indices,
    _selection_stability,
    _verified_final_normalization,
)


class WorkerTests(unittest.TestCase):
    def test_generic_hook_module_classification(self) -> None:
        self.assertEqual(_layer_index("model.layers.12.self_attn"), 12)
        self.assertEqual(_module_category("model.layers.12.self_attn"), "attention")
        self.assertEqual(_module_category("transformer.h.4.mlp"), "mlp")
        self.assertEqual(_module_category("model.layers.12"), "residual")
        self.assertIsNone(_module_category("model.layers.12.input_layernorm"))

    def test_signed_component_attribution_summary(self) -> None:
        summary = _attribution_summary(
            [
                {
                    "layer": 0,
                    "residPre": {"dla": 1.0, "norm": 4.0},
                    "attentionWrite": {"dla": 0.5, "norm": 2.0},
                    "mlpWrite": {"dla": -0.25, "norm": 1.0},
                },
                {
                    "layer": 1,
                    "residPre": {"dla": 1.25, "norm": 4.5},
                    "attentionWrite": {"dla": 0.75, "norm": 2.5},
                    "mlpWrite": None,
                },
            ],
            target_logit=3.0,
        )
        self.assertEqual([item["id"] for item in summary["components"]], ["embedding", "attention.0", "mlp.0", "attention.1"])
        self.assertAlmostEqual(summary["capturedRawSum"], 2.0)
        self.assertAlmostEqual(summary["normalizationAndBiasGap"], 1.0)
        self.assertAlmostEqual(summary["positiveTotal"], 2.25)
        self.assertAlmostEqual(summary["negativeTotal"], -0.25)
        self.assertAlmostEqual(sum(abs(item["shareOfAbsoluteMass"]) for item in summary["components"]), 1.0)

    def test_metric_builder_and_root_cause_stability_contracts(self) -> None:
        metric = _normalise_metric(
            {"kind": "custom_token_groups", "positiveTokens": [" yes"], "negativeTokens": [" no"]}
        )
        self.assertEqual(metric["positiveTokens"], [" yes"])
        stability = _selection_stability(
            [
                {"id": "attention.0", "acdcEffect": 0.03},
                {"id": "mlp.0", "acdcEffect": 0.015},
                {"id": "attention.1", "acdcEffect": 0.004},
            ],
            0.01,
        )
        self.assertEqual([item["retained"] for item in stability["thresholds"]], [2, 2, 1])
        self.assertGreaterEqual(stability["score"], 0)
        self.assertLessEqual(stability["score"], 1)

    def test_metric_value_identity_ignores_labels_but_not_targets(self) -> None:
        first = _normalise_metric({"kind": "target_probability", "name": "Probability", "targetToken": " yes"})
        renamed = _normalise_metric({"kind": "target_probability", "name": "Success", "targetToken": " yes"})
        changed = _normalise_metric({"kind": "target_probability", "name": "Probability", "targetToken": " no"})
        self.assertEqual(_metric_value_identity(first), _metric_value_identity(renamed))
        self.assertNotEqual(_metric_value_identity(first), _metric_value_identity(changed))

    def test_worker_rejects_an_impossible_checkpoint_before_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint needs roughly"):
            _preflight_model_size(object(), {"checkpointBytes": 10**15}, "cpu", "none")

    def test_generation_and_logit_lens_limits_are_explicit(self) -> None:
        generation = _normalise_generation_settings({
            "maxNewTokens": 12,
            "doSample": True,
            "temperature": 0.7,
            "topP": 0.9,
            "topK": 40,
        })
        self.assertEqual(generation["maxNewTokens"], 12)
        self.assertEqual(generation["temperature"], 0.7)
        self.assertEqual(_normalise_generation_settings({"doSample": False})["temperature"], 0.0)
        self.assertEqual(_normalise_lens_settings({"enabled": True, "maxStages": 16})["maxStages"], 16)
        self.assertEqual(_sample_stage_indices(5, 8), [0, 1, 2, 3, 4])
        sampled = _sample_stage_indices(33, 8)
        self.assertEqual((sampled[0], sampled[-1]), (0, 32))
        self.assertEqual(len(sampled), 8)
        with self.assertRaisesRegex(ValueError, "Generation length"):
            _normalise_generation_settings({"maxNewTokens": 100})

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is installed only in the worker environment")
    def test_logit_lens_keeps_inference_tensors_in_inference_mode(self) -> None:
        import torch

        class Tokenizer:
            @staticmethod
            def convert_ids_to_tokens(token_ids):
                return [f"token-{token_id}" for token_id in token_ids]

            @staticmethod
            def decode(token_ids):
                return "".join(f"<{token_id}>" for token_id in token_ids)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(4)
                self.lm_head = torch.nn.Linear(4, 7, bias=False)

            def get_output_embeddings(self):
                return self.lm_head

        model = Model().eval()
        normalization_calls = {}
        handles = _register_normalization_trace(model, normalization_calls)
        with torch.inference_mode():
            hidden_states = [torch.randn(1, 2, 4), model.norm(torch.randn(1, 2, 4))]
            final_logits = model.lm_head(hidden_states[-1][:, -1, :])[0]
        for handle in handles:
            handle.remove()
        verified_normalization = _verified_final_normalization(normalization_calls, hidden_states[-1])
        timeline = _logit_lens_timeline(
            model,
            Tokenizer(),
            hidden_states,
            final_logits,
            1,
            {"enabled": True, "maxStages": 24, "topK": 3},
            verified_normalization,
        )
        self.assertTrue(timeline["available"])
        self.assertEqual(timeline["method"], "normalized-logit-lens")
        self.assertEqual(len(timeline["stages"]), 2)
        self.assertEqual(timeline["stages"][-1]["label"], "Model output")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is installed only in the worker environment")
    def test_logit_lens_supports_tied_embedding_output_modules(self) -> None:
        import torch
        import torch.nn.functional as functional

        class Tokenizer:
            @staticmethod
            def convert_ids_to_tokens(token_ids):
                return [str(token_id) for token_id in token_ids]

            @staticmethod
            def decode(token_ids):
                return " ".join(str(token_id) for token_id in token_ids)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(4)
                self.embedding = torch.nn.Embedding(7, 4)

            def get_output_embeddings(self):
                return self.embedding

        model = Model().eval()
        normalization_calls = {}
        handles = _register_normalization_trace(model, normalization_calls)
        with torch.inference_mode():
            hidden_states = [torch.randn(1, 2, 4), model.norm(torch.randn(1, 2, 4))]
            final_logits = functional.linear(hidden_states[-1][:, -1, :], model.embedding.weight)[0]
        for handle in handles:
            handle.remove()
        timeline = _logit_lens_timeline(
            model,
            Tokenizer(),
            hidden_states,
            final_logits,
            1,
            {"enabled": True, "maxStages": 24, "topK": 3},
            _verified_final_normalization(normalization_calls, hidden_states[-1]),
        )
        self.assertTrue(timeline["available"])
        self.assertEqual(timeline["outputProjection"], "get_output_embeddings()")

    def test_causal_sweep_axes_normalise_negative_positions_and_cap_work(self) -> None:
        axes = _normalise_sweep_axes(
            {"kind": "mlp", "method": "mean", "layers": [2, 0, 2], "positions": [-1, 1]},
            layer_count=4,
            token_count=6,
        )
        self.assertEqual(axes["layers"], [0, 2])
        self.assertEqual(axes["positions"], [1, 5])
        self.assertEqual(axes["kind"], "mlp")
        with self.assertRaisesRegex(ValueError, "128"):
            _normalise_sweep_axes(
                {"layers": list(range(32)), "positions": list(range(5))},
                layer_count=32,
                token_count=5,
            )


if __name__ == "__main__":
    unittest.main()
