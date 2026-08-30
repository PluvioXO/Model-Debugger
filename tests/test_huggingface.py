from __future__ import annotations

import json
import unittest
from email.message import Message
from unittest.mock import patch

from refusalscope.http_client import HTTPResponse
from refusalscope.huggingface import (
    ModelFile,
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


if __name__ == "__main__":
    unittest.main()
