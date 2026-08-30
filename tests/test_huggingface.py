from __future__ import annotations

import json
import unittest
from email.message import Message
from unittest.mock import patch

from refusalscope.http_client import HTTPResponse
from refusalscope.huggingface import (
    ModelFile,
    _checkpoint_file_inventory,
    _pytorch_manifest,
    inspect_account,
    inspect_model,
    inspect_safetensors_file,
    normalise_dtype,
    weight_file_names,
)


class HuggingFaceTests(unittest.TestCase):
    def test_dtype_and_weight_file_selection(self) -> None:
        self.assertEqual(normalise_dtype("BF16"), "bfloat16")
        files = [
            ModelFile("adapter.safetensors", 1),
            ModelFile("optimizer.safetensors", 1),
            ModelFile("model-00002-of-00002.safetensors", 1),
            ModelFile("model-00001-of-00002.safetensors", 1),
        ]
        self.assertEqual(weight_file_names(files), [files[3].name, files[2].name])

    def test_checkpoint_formats_and_pytorch_manifests_are_inventory_only(self) -> None:
        files = [
            ModelFile("pytorch_model-00001-of-00002.bin", 10),
            ModelFile("pytorch_model-00002-of-00002.bin", 12),
            ModelFile("adapter_model.safetensors", 3),
            ModelFile("model.Q4_K_M.gguf", 9),
            ModelFile("optimizer.pt", 99),
        ]
        inventory = _checkpoint_file_inventory(files)
        self.assertEqual(len(inventory["pytorch"]), 2)
        self.assertEqual(inventory["adapter"], ["adapter_model.safetensors"])
        self.assertEqual(inventory["gguf"], ["model.Q4_K_M.gguf"])
        self.assertEqual(weight_file_names(files), [])
        name, manifest = _pytorch_manifest({
            "pytorch_model.bin.index.json": {
                "data": {"weight_map": {"model.layers.0.weight": "pytorch_model-00001-of-00002.bin"}}
            }
        })
        self.assertEqual(name, "pytorch_model.bin.index.json")
        self.assertIn("model.layers.0.weight", manifest["weight_map"])

    def test_safetensors_header_is_preserved(self) -> None:
        header = {
            "__metadata__": {"format": "pt"},
            "weight": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]},
        }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        body = len(encoded).to_bytes(8, "little") + encoded
        headers = Message()
        headers["Content-Range"] = f"bytes 0-{len(body) - 1}/{len(body) + 12}"
        headers["ETag"] = '"test"'
        prefix = HTTPResponse(206, body, headers)

        tensors, record, tensor_bytes = inspect_safetensors_file(
            "org/model", "main", "model.safetensors", "", prefix
        )

        self.assertEqual(tensors["weight"]["dtype"], "float16")
        self.assertEqual(tensors["weight"]["shape"], [2, 3])
        self.assertEqual(tensors["weight"]["safetensors"]["raw"], header["weight"])
        self.assertEqual(tensors["weight"]["safetensors"]["fileTensorIndex"], 0)
        self.assertEqual(record["metadata"], {"format": "pt"})
        self.assertEqual(record["fileSize"], len(body) + 12)
        self.assertEqual(record["etag"], '"test"')
        self.assertEqual(tensor_bytes, 12)

    @patch("refusalscope.huggingface.fetch_json")
    def test_account_response_is_sanitized(self, fetch_json) -> None:
        fetch_json.return_value = {
            "name": "tester",
            "fullname": "Test User",
            "avatarUrl": "https://example.test/avatar.png",
            "type": "user",
            "isPro": True,
            "email": "private@example.test",
            "orgs": [{"name": "org", "fullname": "Org", "avatarUrl": "https://example.test/org.png", "secret": 1}],
        }
        account = inspect_account("hf_test")
        self.assertEqual(account["avatarUrl"], "https://example.test/avatar.png")
        self.assertNotIn("email", account)
        self.assertNotIn("secret", account["orgs"][0])

    def test_live_tiny_model_inspection(self) -> None:
        graph = inspect_model("hf-internal-testing/tiny-random-LlamaForCausalLM", "main")["graph"]
        self.assertEqual(graph["safetensors"]["fileCount"], 1)
        self.assertEqual(graph["safetensors"]["tensorCount"], 21)
        self.assertEqual(graph["stats"]["parameterTensors"], 21)
        self.assertEqual(len(graph["groups"]), 2)
        self.assertEqual(graph["validation"]["status"], "verified")
        self.assertEqual(graph["forwardTopology"]["residual"], "sequential-pre-norm")
        self.assertEqual(graph["forwardTopology"]["positionKind"], "rotary")
        node_ids = {node["id"] for node in graph["nodes"]}
        self.assertTrue({"l0_position", "l1_position", "final_norm", "lm_head"} <= node_ids)
        self.assertNotIn("unmapped_tensors", node_ids)
        self.assertTrue(graph["tensorOrdering"]["automatic"])
        self.assertEqual(graph["tensorOrdering"]["checkpointLocated"], 21)

    def test_live_pytorch_only_model_opens_as_configuration_scaffold(self) -> None:
        graph = inspect_model("hf-internal-testing/tiny-random-GPTNeoXForCausalLM", "main")["graph"]
        self.assertEqual(graph["resolver"]["tier"], "configuration-scaffold")
        self.assertEqual(graph["resolver"]["format"], "pytorch")
        self.assertEqual(graph["resolver"]["checkpointFiles"], ["pytorch_model.bin"])
        self.assertIsNone(graph["stats"]["checkpointTensors"])
        self.assertIsNone(graph["stats"]["checkpointElements"])
        self.assertEqual(graph["validation"]["status"], "partial")
        self.assertEqual(len(graph["groups"]), 5)
        checkpoint = next(node for node in graph["nodes"] if node["id"] == "checkpoint_metadata")
        self.assertEqual(checkpoint["name"], "Repository capability")


if __name__ == "__main__":
    unittest.main()
