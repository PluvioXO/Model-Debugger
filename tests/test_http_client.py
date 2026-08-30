from __future__ import annotations

import unittest

from refusalscope.http_client import http_get


class HTTPClientTests(unittest.TestCase):
    def test_live_https_request(self) -> None:
        response = http_get(
            "https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/main/config.json"
        )
        self.assertIsNone(response.error)
        self.assertEqual(response.status, 200)
        self.assertIn("hidden_size", response.text)


if __name__ == "__main__":
    unittest.main()
