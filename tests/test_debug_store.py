from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from refusalscope.debug_store import DebugStore, DebugStoreError


class DebugStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "debug.sqlite3"
        self.store = DebugStore(self.path)

    def tearDown(self) -> None:
        self.store.close()
        self.directory.cleanup()

    def test_case_round_trip_persists_and_lists_summary(self) -> None:
        created = self.store.create_case(
            {
                "name": "Capital failure",
                "model": {"modelId": "test/model", "revision": "abc123"},
                "failure": {"prompt": "The capital of France is"},
                "control": {"prompt": "France's capital is"},
                "expected": {"text": " Paris"},
                "experiments": [{"id": "run-1"}],
            }
        )
        self.assertEqual(len(created["id"]), 32)
        self.assertEqual(self.store.get_case(created["id"])["expected"]["text"], " Paris")
        summary = self.store.list_cases()[0]
        self.assertEqual(summary["modelId"], "test/model")
        self.assertEqual(summary["experimentCount"], 1)

        self.store.close()
        self.store = DebugStore(self.path)
        self.assertEqual(self.store.get_case(created["id"])["name"], "Capital failure")

    def test_replace_preserves_identity_and_delete_removes_case(self) -> None:
        created = self.store.create_case({"failure": {"prompt": "Unexpected output"}})
        replaced = self.store.replace_case(
            created["id"],
            {"name": "Renamed", "failure": {"prompt": "Unexpected output"}, "status": "verified"},
        )
        self.assertEqual(replaced["id"], created["id"])
        self.assertEqual(replaced["createdAt"], created["createdAt"])
        self.assertEqual(replaced["status"], "verified")
        self.assertTrue(self.store.delete_case(created["id"]))
        self.assertIsNone(self.store.get_case(created["id"]))

    def test_invalid_json_and_oversized_names_are_rejected(self) -> None:
        with self.assertRaises(DebugStoreError):
            self.store.create_case({"value": float("nan")})
        with self.assertRaises(DebugStoreError):
            self.store.create_case({"name": "x" * 181})


if __name__ == "__main__":
    unittest.main()
