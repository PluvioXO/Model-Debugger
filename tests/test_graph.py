from __future__ import annotations

import unittest

from refusalscope.graph import GraphError, NODE_HEIGHT, NODE_WIDTH, TENSOR_STACK_MAX_EXTENT, build_model_graph


class GraphTests(unittest.TestCase):
    def test_nested_decoder_graph_contract(self) -> None:
        payload = {
            "modelId": "org/composite-model",
            "revision": "main",
            "sha": "abcdef123456",
            "config": {
                "model_type": "vision_text_model",
                "architectures": ["CompositeForConditionalGeneration"],
                "vision_config": {"num_hidden_layers": 8, "hidden_size": 128},
                "text_config": {
                    "model_type": "qwen2",
                    "num_hidden_layers": 2,
                    "hidden_size": 16,
                    "vocab_size": 100,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "rope_theta": 10000.0,
                    "hidden_act": "silu",
                    "rms_norm_eps": 0.00001,
                },
            },
            "tensors": {
                "vision_model.layers.0.attn.weight": self.tensor([4, 4]),
                "language_model.embed_tokens.weight": self.tensor([100, 16]),
                "language_model.layers.0.self_attn.q_proj.weight": self.tensor([16, 16]),
                "language_model.layers.1.self_attn.q_proj.weight": self.tensor([16, 16]),
                "language_model.norm.weight": self.tensor([16]),
                "lm_head.weight": self.tensor([100, 16]),
            },
            "files": ["model.safetensors"],
            "safetensors": {"files": [], "tensorCount": 6},
            "hub": {
                "author": "org",
                "siblings": [
                    {"rfilename": "config.json", "size": 123},
                    {"rfilename": "model.safetensors", "size": 456},
                ],
            },
            "artifacts": {"config.json": {"format": "json", "data": {"model_type": "qwen2"}}},
            "artifactInspection": {"fetchedCount": 1, "skipped": []},
        }

        graph = build_model_graph(payload)
        nodes = {node["id"]: node for node in graph["nodes"]}

        self.assertEqual(graph["mode"], "flow")
        self.assertEqual(len(graph["groups"]), 2)
        self.assertEqual(graph["resolvedLayerFamily"]["prefix"], "language_model.layers")
        self.assertEqual(nodes["l1_qkv"]["position"]["column"], 3)
        self.assertEqual(nodes["l1_mlp"]["position"]["column"], 4)
        inspector = nodes["l0_qkv"]["inspector"]
        self.assertEqual(inspector["config"]["num_attention_heads"], 4)
        self.assertTrue(inspector["connections"])
        self.assertTrue(inspector["findings"])
        ledger = graph["residualLedger"]
        self.assertEqual(ledger["mode"], "structural")
        self.assertEqual(ledger["metric"]["status"], "not-measured")
        self.assertEqual(len(ledger["states"]), 3)
        self.assertEqual(ledger["states"][1]["equation"], "h1 = h0 + a0 + m0")
        self.assertEqual(
            [write["sourceNodeId"] for write in ledger["states"][2]["writes"]],
            ["l1_output", "l1_mlp"],
        )
        self.assertIsNone(ledger["states"][1]["writes"][0]["directLogitAttribution"])
        self.assertEqual(nodes["l0_mlp_residual"]["inspector"]["residualLedger"]["role"], "state")
        self.assertEqual(nodes["l0_output"]["inspector"]["residualLedger"]["role"], "write")
        self.assertEqual(nodes["l0_attn_residual"]["inspector"]["residualLedger"]["role"], "write")
        self.assertEqual(graph["stats"]["parameterTensors"], 6)
        self.assertEqual(graph["validation"]["status"], "verified")
        self.assertEqual(graph["forwardTopology"]["residual"], "sequential-pre-norm")
        self.assertIn("Q [B, 4, T, 4]", nodes["l0_qkv"]["shape"])
        self.assertIn("K/V [B, 2, T, 4]", nodes["l0_qkv"]["shape"])
        self.assertIn("l0_position", nodes)
        self.assertIn("broadcast_G(K′)", nodes["l0_attention"]["formula"])
        self.assertIn("cache remains at Hkv", nodes["l0_attention"]["description"])
        self.assertIn(("l0_position", "l0_kv_cache"), self.edge_pairs(graph))
        self.assertTrue(graph["edges"][0]["path"])
        self.assertTrue(any(" L " in edge["path"] for edge in graph["edges"] if edge["feedback"]))

    def test_config_declared_layers_are_retained_without_indexed_weights(self) -> None:
        graph = build_model_graph({
            "modelId": "org/non-indexed",
            "revision": "main",
            "sha": None,
            "config": {"model_type": "custom_lm", "num_hidden_layers": 12, "hidden_size": 8, "vocab_size": 32},
            "tensors": {
                "token_embedding.weight": {"shape": [32, 8], "dtype": "float32"},
                "combined_decoder.weight": {"shape": [8, 8], "dtype": "float32"},
            },
            "files": ["model.safetensors"],
            "safetensors": {"files": []},
            "hub": {"siblings": []},
            "artifacts": {},
            "artifactInspection": {},
        })
        nodes = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(len(graph["groups"]), 12)
        self.assertTrue(graph["resolvedLayerFamily"]["predicted"])
        self.assertTrue(all(not group["evidence"]["checkpointMapped"] for group in graph["groups"]))
        self.assertEqual(len(nodes["unmapped_tensors"]["tensors"]), 2)

    def test_gpt2_absolute_positions_and_causal_mask_are_mapped_correctly(self) -> None:
        graph = build_model_graph({
            "modelId": "openai-community/gpt2",
            "revision": "main",
            "sha": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            "config": {
                "model_type": "gpt2", "n_layer": 1, "n_embd": 8, "n_vocab": 32,
                "n_head": 2, "n_positions": 16, "layer_norm_epsilon": 1e-5,
                "activation_function": "gelu_new", "tie_word_embeddings": True,
            },
            "tensors": {
                "transformer.wte.weight": self.tensor([32, 8]),
                "transformer.wpe.weight": self.tensor([16, 8]),
                "transformer.h.0.ln_1.weight": self.tensor([8]),
                "transformer.h.0.ln_1.bias": self.tensor([8]),
                "transformer.h.0.attn.c_attn.weight": self.tensor([8, 24]),
                "transformer.h.0.attn.c_proj.weight": self.tensor([8, 8]),
                "transformer.h.0.attn.bias": self.tensor([1, 1, 16, 16]),
                "transformer.h.0.ln_2.weight": self.tensor([8]),
                "transformer.h.0.mlp.c_fc.weight": self.tensor([8, 32]),
                "transformer.h.0.mlp.c_proj.weight": self.tensor([32, 8]),
                "transformer.ln_f.weight": self.tensor([8]),
            },
            "safetensors": {"files": [], "tensorCount": 11},
            "hub": {"siblings": []}, "artifacts": {}, "artifactInspection": {},
        })
        nodes = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(graph["forwardTopology"]["positionKind"], "absolute")
        self.assertIn("position_embedding", nodes)
        self.assertNotIn("l0_position", nodes)
        self.assertEqual(nodes["residual_0"]["formula"], "h₀ = e_tok + e_pos")
        self.assertEqual(nodes["position_embedding"]["tensors"][0]["name"], "transformer.wpe.weight")
        mask = next(tensor for tensor in nodes["l0_attention"]["tensors"] if tensor["name"].endswith("attn.bias"))
        self.assertEqual(mask["storageRole"], "buffer")
        self.assertEqual(mask["order"]["parameterKind"], "Buffer")
        self.assertNotIn("l0_other", nodes)
        self.assertEqual(graph["stats"]["bufferTensors"], 1)
        self.assertEqual(graph["stats"]["recognizedBufferElements"], 256)
        self.assertEqual(graph["stats"]["checkpointElements"], sum(self.count(node) for node in graph["nodes"]))

    def test_gptj_parallel_shared_norm_uses_one_branch_input(self) -> None:
        graph = build_model_graph(self.single_layer_payload(
            model_type="gptj",
            config={"n_head": 2, "rotary_dim": 4},
            prefix="transformer.h",
            tensors={
                "ln_1.weight": [8],
                "attn.q_proj.weight": [8, 8], "attn.k_proj.weight": [8, 8], "attn.v_proj.weight": [8, 8],
                "attn.out_proj.weight": [8, 8], "mlp.fc_in.weight": [32, 8], "mlp.fc_out.weight": [8, 32],
            },
        ))
        nodes = {node["id"]: node for node in graph["nodes"]}
        edges = self.edge_pairs(graph)
        self.assertEqual(graph["forwardTopology"]["residual"], "parallel-shared-norm")
        self.assertNotIn("l0_attn_residual", nodes)
        self.assertNotIn("l0_post_norm", nodes)
        self.assertIn(("l0_input_norm", "l0_qkv"), edges)
        self.assertIn(("l0_input_norm", "l0_mlp"), edges)
        self.assertIn(("residual_0", "l0_mlp_residual"), edges)

    def test_gpt_neox_parallel_residual_uses_separate_branch_norms(self) -> None:
        graph = build_model_graph(self.single_layer_payload(
            model_type="gpt_neox",
            config={"num_attention_heads": 2, "rotary_pct": 0.25, "use_parallel_residual": True},
            prefix="gpt_neox.layers",
            tensors={
                "input_layernorm.weight": [8], "post_attention_layernorm.weight": [8],
                "attention.query_key_value.weight": [24, 8], "attention.dense.weight": [8, 8],
                "mlp.dense_h_to_4h.weight": [32, 8], "mlp.dense_4h_to_h.weight": [8, 32],
            },
        ))
        nodes = {node["id"]: node for node in graph["nodes"]}
        edges = self.edge_pairs(graph)
        self.assertEqual(graph["forwardTopology"]["residual"], "parallel-dual-norm")
        self.assertIn("l0_post_norm", nodes)
        self.assertNotIn("l0_attn_residual", nodes)
        self.assertIn(("residual_0", "l0_input_norm"), edges)
        self.assertIn(("residual_0", "l0_post_norm"), edges)
        self.assertIn(("l0_output", "l0_mlp_residual"), edges)
        self.assertIn(("l0_mlp", "l0_mlp_residual"), edges)

    def test_opt_post_norm_preserves_local_residual_semantics(self) -> None:
        graph = build_model_graph({
            "modelId": "org/opt-fixture", "revision": "main", "sha": "abc123",
            "config": {
                "model_type": "opt", "num_hidden_layers": 1, "hidden_size": 8, "vocab_size": 32,
                "num_attention_heads": 2, "do_layer_norm_before": False, "tie_word_embeddings": True,
            },
            "tensors": {
                "model.decoder.embed_tokens.weight": self.tensor([32, 8]),
                "model.decoder.embed_positions.weight": self.tensor([18, 8]),
                "model.decoder.layers.0.self_attn_layer_norm.weight": self.tensor([8]),
                "model.decoder.layers.0.self_attn.q_proj.weight": self.tensor([8, 8]),
                "model.decoder.layers.0.self_attn.k_proj.weight": self.tensor([8, 8]),
                "model.decoder.layers.0.self_attn.v_proj.weight": self.tensor([8, 8]),
                "model.decoder.layers.0.self_attn.out_proj.weight": self.tensor([8, 8]),
                "model.decoder.layers.0.final_layer_norm.weight": self.tensor([8]),
                "model.decoder.layers.0.fc1.weight": self.tensor([32, 8]),
                "model.decoder.layers.0.fc2.weight": self.tensor([8, 32]),
            },
            "safetensors": {"files": [], "tensorCount": 10},
            "hub": {"siblings": []}, "artifacts": {}, "artifactInspection": {},
        })
        nodes = {node["id"]: node for node in graph["nodes"]}
        edges = self.edge_pairs(graph)
        self.assertEqual(graph["forwardTopology"]["residual"], "sequential-post-norm")
        self.assertEqual(graph["residualLedger"]["mode"], "local-residual")
        self.assertEqual(graph["residualLedger"]["states"][1]["equation"], "h1 = Norm(Norm(h0 + a0) + m0)")
        self.assertNotIn("l0_post_norm", nodes)
        self.assertIn(("residual_0", "l0_qkv"), edges)
        self.assertNotIn(("residual_0", "l0_mlp_residual"), edges)
        self.assertEqual(nodes["l0_mlp_residual"]["tensors"][0]["name"], "model.decoder.layers.0.final_layer_norm.weight")

    def test_tensor_records_follow_forward_function_order(self) -> None:
        graph = build_model_graph({
            "modelId": "org/ordered-model",
            "revision": "main",
            "sha": "abcdef123456",
            "config": {"model_type": "qwen2", "num_hidden_layers": 1, "hidden_size": 8, "vocab_size": 32},
            "tensors": {
                "model.embed_tokens.weight": self.tensor([32, 8]),
                "model.layers.0.self_attn.v_proj.weight": self.tensor([8, 8]),
                "model.layers.0.self_attn.k_proj.weight": self.tensor([8, 8]),
                "model.layers.0.self_attn.q_proj.bias": self.tensor([8]),
                "model.layers.0.self_attn.q_proj.weight": self.tensor([8, 8]),
                "model.layers.0.self_attn.o_proj.weight": self.tensor([8, 8]),
                "model.layers.0.input_layernorm.weight": self.tensor([8]),
                "model.layers.0.post_attention_layernorm.weight": self.tensor([8]),
                "model.layers.0.mlp.down_proj.weight": self.tensor([8, 16]),
                "model.layers.0.mlp.up_proj.weight": self.tensor([16, 8]),
                "model.layers.0.mlp.gate_proj.weight": self.tensor([16, 8]),
                "model.norm.weight": self.tensor([8]),
                "lm_head.weight": self.tensor([32, 8]),
            },
            "safetensors": {"files": [], "tensorCount": 13},
            "hub": {"siblings": []},
            "artifacts": {},
            "artifactInspection": {},
        })
        nodes = {node["id"]: node for node in graph["nodes"]}
        qkv = nodes["l0_qkv"]["tensors"]
        self.assertEqual(
            [tensor["order"]["semanticRole"] for tensor in qkv],
            ["Query projection", "Query projection", "Key projection", "Value projection"],
        )
        self.assertTrue(qkv[0]["name"].endswith("q_proj.weight"))
        self.assertTrue(qkv[1]["name"].endswith("q_proj.bias"))
        self.assertEqual([tensor["order"]["operationIndex"] for tensor in qkv], [0, 1, 2, 3])
        self.assertEqual(
            [tensor["order"]["semanticRole"] for tensor in nodes["l0_mlp"]["tensors"]],
            ["Gate projection", "MLP expansion projection", "MLP contraction projection"],
        )

    def test_unknown_architecture_tensor_names_receive_automatic_fallback_order(self) -> None:
        graph = build_model_graph({
            "modelId": "org/generic-gpt",
            "revision": "main",
            "sha": "abcdef123456",
            "config": {"model_type": "gpt_variant", "n_layer": 1, "n_embd": 8, "n_vocab": 32},
            "tensors": {
                "transformer.wte.weight": self.tensor([32, 8]),
                "transformer.h.0.ln_1.weight": self.tensor([8]),
                "transformer.h.0.attn.c_attn.weight": self.tensor([8, 24]),
                "transformer.h.0.attn.c_proj.weight": self.tensor([8, 8]),
                "transformer.h.0.ln_2.weight": self.tensor([8]),
                "transformer.h.0.mlp.c_proj.weight": self.tensor([8, 16]),
                "transformer.h.0.mlp.mystery_adapter.weight": self.tensor([16, 16]),
                "transformer.h.0.mlp.c_fc.weight": self.tensor([16, 8]),
                "transformer.ln_f.weight": self.tensor([8]),
                "lm_head.weight": self.tensor([32, 8]),
            },
            "safetensors": {"files": [], "tensorCount": 10},
            "hub": {"siblings": []},
            "artifacts": {},
            "artifactInspection": {},
        })
        nodes = {node["id"]: node for node in graph["nodes"]}
        mlp = nodes["l0_mlp"]["tensors"]
        self.assertEqual(mlp[0]["order"]["semanticRole"], "MLP expansion projection")
        self.assertEqual(mlp[1]["order"]["semanticRole"], "MLP contraction projection")
        self.assertEqual(mlp[2]["order"]["semanticRole"], "Mystery adapter parameter")
        self.assertEqual(mlp[2]["order"]["semanticConfidence"], "path-derived")
        self.assertTrue(all(tensor["order"]["automatic"] for tensor in mlp))
        self.assertTrue(graph["tensorOrdering"]["automatic"])
        self.assertGreaterEqual(graph["tensorOrdering"]["pathDerived"], 1)

    def test_tensor_stacks_stay_inside_group_and_graph_bounds(self) -> None:
        graph = build_model_graph({
            "modelId": "org/stacked-model",
            "revision": "main",
            "sha": "abcdef123456",
            "config": {"model_type": "custom_lm", "num_hidden_layers": 1, "hidden_size": 8, "vocab_size": 32},
            "tensors": {
                "model.embed_tokens.weight": self.tensor([32, 8]),
                "model.layers.0.mlp.experts.weight": self.tensor([2, 4, 8, 8]),
                "lm_head.weight": self.tensor([2, 4, 8, 8]),
            },
            "safetensors": {"files": [], "tensorCount": 3},
            "hub": {"siblings": []},
            "artifacts": {},
            "artifactInspection": {},
        })
        nodes = {node["id"]: node for node in graph["nodes"]}
        group = graph["groups"][0]
        subgroup = next(item for item in group["subgroups"] if item["kind"] == "mlp")
        stacked_node = nodes["l0_mlp"]
        node_right = stacked_node["layout"]["x"] + NODE_WIDTH + TENSOR_STACK_MAX_EXTENT
        node_bottom = stacked_node["layout"]["y"] + NODE_HEIGHT + TENSOR_STACK_MAX_EXTENT

        self.assertLessEqual(node_right, subgroup["bounds"]["x"] + subgroup["bounds"]["width"])
        self.assertLessEqual(node_bottom, subgroup["bounds"]["y"] + subgroup["bounds"]["height"])
        self.assertLessEqual(node_right, group["bounds"]["x"] + group["bounds"]["width"])
        self.assertLessEqual(node_bottom, group["bounds"]["y"] + group["bounds"]["height"])

        output_node = nodes["lm_head"]
        self.assertLessEqual(
            output_node["layout"]["x"] + NODE_WIDTH + TENSOR_STACK_MAX_EXTENT,
            graph["layout"]["bounds"]["width"],
        )
        self.assertLessEqual(
            output_node["layout"]["y"] + NODE_HEIGHT + TENSOR_STACK_MAX_EXTENT,
            graph["layout"]["bounds"]["height"],
        )

    def test_missing_layer_evidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(GraphError, "Could not identify decoder layers"):
            build_model_graph({
                "modelId": "org/invalid",
                "revision": "main",
                "config": {},
                "tensors": {},
                "hub": {},
            })

    @staticmethod
    def tensor(shape: list[int]) -> dict:
        return {
            "shape": shape,
            "dtype": "float16",
            "safetensors": {"file": "model.safetensors"},
        }

    @staticmethod
    def edge_pairs(graph: dict) -> set[tuple[str, str]]:
        return {(edge["from"], edge["to"]) for edge in graph["edges"]}

    @staticmethod
    def count(node: dict) -> int:
        return sum(tensor["count"] for tensor in node["tensors"])

    def single_layer_payload(self, model_type: str, config: dict, prefix: str, tensors: dict[str, list[int]]) -> dict:
        inventory = {"model.embed_tokens.weight": self.tensor([32, 8])}
        inventory.update({f"{prefix}.0.{name}": self.tensor(shape) for name, shape in tensors.items()})
        inventory["model.norm.weight"] = self.tensor([8])
        return {
            "modelId": f"org/{model_type}-fixture", "revision": "main", "sha": "abc123",
            "config": {"model_type": model_type, "num_hidden_layers": 1, "hidden_size": 8, "vocab_size": 32, **config},
            "tensors": inventory,
            "safetensors": {"files": [], "tensorCount": len(inventory)},
            "hub": {"siblings": []}, "artifacts": {}, "artifactInspection": {},
        }


if __name__ == "__main__":
    unittest.main()
