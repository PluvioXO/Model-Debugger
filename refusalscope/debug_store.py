"""SQLite persistence for reproducible ModelDebugger cases and experiments."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MAX_CASE_BYTES = 8 * 1024 * 1024
MAX_CASE_NAME = 180


class DebugStoreError(ValueError):
    """A debug-case request could not be validated or persisted."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_copy(value: Any) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise DebugStoreError(f"Debug case must contain valid JSON: {error}") from error
    if len(encoded.encode("utf-8")) > MAX_CASE_BYTES:
        raise DebugStoreError("Debug case exceeds the 8 MB local persistence limit")
    return json.loads(encoded), encoded


def _case_name(payload: dict[str, Any]) -> str:
    name = str(payload.get("name", "")).strip()
    if not name:
        selected = payload.get("selected") if isinstance(payload.get("selected"), dict) else payload.get("failure", {})
        prompt = str(selected.get("prompt", "")).strip()
        name = prompt[:72] if prompt else "Untitled debug case"
    if len(name) > MAX_CASE_NAME:
        raise DebugStoreError(f"Debug case names are limited to {MAX_CASE_NAME} characters")
    return name


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    selected = record.get("selected") if isinstance(record.get("selected"), dict) else record.get("failure") if isinstance(record.get("failure"), dict) else {}
    model = record.get("model") if isinstance(record.get("model"), dict) else {}
    experiments = record.get("experiments") if isinstance(record.get("experiments"), list) else []
    verification = record.get("verification") if isinstance(record.get("verification"), dict) else {}
    return {
        "id": record["id"],
        "name": record["name"],
        "createdAt": record["createdAt"],
        "updatedAt": record["updatedAt"],
        "modelId": str(model.get("modelId", "")),
        "revision": str(model.get("revision", "")),
        "promptPreview": str(selected.get("prompt", ""))[:160],
        "status": str(record.get("status", "open")),
        "experimentCount": len(experiments),
        "verificationStatus": str(verification.get("status", "not-run")),
    }


class DebugStore:
    """Thread-safe JSON document store backed by a small local SQLite database."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock, self._connection:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS debug_cases (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_id TEXT NOT NULL DEFAULT '',
                    revision TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS debug_cases_updated ON debug_cases(updated_at DESC)"
            )
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def list_cases(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM debug_cases ORDER BY updated_at DESC, id ASC"
            ).fetchall()
        return [_summary(json.loads(row["payload"])) for row in rows]

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        if not _valid_identifier(case_id):
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM debug_cases WHERE id = ?", (case_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def create_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        clean, _encoded = _json_copy(payload)
        if not isinstance(clean, dict):
            raise DebugStoreError("Debug case must be a JSON object")
        now = _timestamp()
        record = {
            **clean,
            "id": uuid.uuid4().hex,
            "name": _case_name(clean),
            "status": str(clean.get("status", "open"))[:40] or "open",
            "schemaVersion": SCHEMA_VERSION,
            "createdAt": now,
            "updatedAt": now,
        }
        self._write(record, insert=True)
        return record

    def replace_case(self, case_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        existing = self.get_case(case_id)
        if existing is None:
            return None
        clean, _encoded = _json_copy(payload)
        if not isinstance(clean, dict):
            raise DebugStoreError("Debug case must be a JSON object")
        record = {
            **clean,
            "id": case_id,
            "name": _case_name(clean),
            "status": str(clean.get("status", existing.get("status", "open")))[:40] or "open",
            "schemaVersion": SCHEMA_VERSION,
            "createdAt": existing["createdAt"],
            "updatedAt": _timestamp(),
        }
        self._write(record, insert=False)
        return record

    def delete_case(self, case_id: str) -> bool:
        if not _valid_identifier(case_id):
            return False
        with self._lock, self._connection:
            cursor = self._connection.execute("DELETE FROM debug_cases WHERE id = ?", (case_id,))
        return cursor.rowcount > 0

    def _write(self, record: dict[str, Any], *, insert: bool) -> None:
        clean, encoded = _json_copy(record)
        model = clean.get("model") if isinstance(clean.get("model"), dict) else {}
        values = (
            clean["id"],
            clean["name"],
            str(model.get("modelId", "")),
            str(model.get("revision", "")),
            clean["status"],
            clean["createdAt"],
            clean["updatedAt"],
            encoded,
        )
        statement = (
            "INSERT INTO debug_cases(id,name,model_id,revision,status,created_at,updated_at,payload) "
            "VALUES(?,?,?,?,?,?,?,?)"
            if insert
            else "UPDATE debug_cases SET name=?, model_id=?, revision=?, status=?, created_at=?, updated_at=?, payload=? WHERE id=?"
        )
        if not insert:
            values = (*values[1:], values[0])
        try:
            with self._lock, self._connection:
                self._connection.execute(statement, values)
        except sqlite3.IntegrityError as error:
            raise DebugStoreError("Could not save this debug case") from error


def _valid_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{32}", value))
