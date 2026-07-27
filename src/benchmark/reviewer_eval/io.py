"""Canonical, crash-safe persistence primitives for reviewer evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary.replace(destination)
            _fsync_directory(destination.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def atomic_write_json(path: str | Path, payload: object) -> None:
    _atomic_write_bytes(Path(path), (canonical_json(payload) + "\n").encode("utf-8"))


def atomic_write_jsonl(path: str | Path, rows: Iterable[object]) -> None:
    """Atomically replace a JSONL artifact with canonical object rows."""
    _atomic_write_bytes(
        Path(path),
        "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8"),
    )


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL records in {source} must be JSON objects")
    return rows


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class JsonlLedger:
    """Append-only JSONL ledger with a stable resume index.

    The adjacent ``run_manifest.json`` records every repaired truncated tail
    before that tail is excluded from the ledger's resume index.
    """

    def __init__(self, path: str | Path, *, key_fields: tuple[str, ...]) -> None:
        if not key_fields:
            raise ValueError("key_fields cannot be empty")
        self.path = Path(path)
        self.key_fields = key_fields
        self._payload_by_key: dict[tuple[Any, ...], str] = {}
        self._reload_under_lock()

    @property
    def _lock_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.lock")

    @property
    def _manifest_path(self) -> Path:
        return self.path.with_name("run_manifest.json")

    def _key_for(self, record: Mapping[str, object]) -> tuple[Any, ...]:
        missing = [field for field in self.key_fields if field not in record]
        if missing:
            raise ValueError(f"ledger record is missing key fields: {', '.join(missing)}")
        return tuple(record[field] for field in self.key_fields)

    def _reload_under_lock(self) -> None:
        with _exclusive_lock(self._lock_path):
            records = self._read_and_repair_locked()
            self._payload_by_key = self._index_records(records)

    def _index_records(self, records: list[dict[str, object]]) -> dict[tuple[Any, ...], str]:
        indexed: dict[tuple[Any, ...], str] = {}
        for record in records:
            key = self._key_for(record)
            payload = canonical_json(record)
            previous = indexed.get(key)
            if previous is not None and previous != payload:
                raise ValueError(f"conflicting payload for duplicate ledger key {key!r}")
            indexed[key] = payload
        return indexed

    def _read_and_repair_locked(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []

        data = self.path.read_bytes()
        lines = data.splitlines(keepends=True)
        records: list[dict[str, object]] = []
        truncated_tail: bytes | None = None
        truncated_index: int | None = None

        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                is_unterminated_final_line = (
                    index == len(lines) - 1 and not line.endswith((b"\n", b"\r"))
                )
                if is_unterminated_final_line:
                    truncated_tail = line
                    truncated_index = index
                    break
                raise ValueError(
                    f"malformed non-final JSONL record in {self.path} at line {index + 1}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL record in {self.path} at line {index + 1} must be an object"
                )
            records.append(record)

        if truncated_tail is not None and truncated_index is not None:
            self._repair_truncated_tail_locked(
                valid_bytes=b"".join(lines[:truncated_index]),
                truncated_tail=truncated_tail,
            )
        return records

    def _repair_truncated_tail_locked(self, *, valid_bytes: bytes, truncated_tail: bytes) -> None:
        corrupt_path = self.path.with_name(f"{self.path.name}.corrupt")
        tail_digest = hashlib.sha256(truncated_tail).hexdigest()
        repair = {
            "kind": "truncated_jsonl_final_line",
            "ledger": self.path.name,
            "corrupt_file": corrupt_path.name,
            "sha256": tail_digest,
        }

        # Preserve evidence first; a retry recognizes the same tail by digest.
        if not corrupt_path.exists():
            _atomic_write_bytes(corrupt_path, truncated_tail)
        elif truncated_tail not in corrupt_path.read_bytes().splitlines():
            with corrupt_path.open("ab") as handle:
                if corrupt_path.stat().st_size:
                    handle.write(b"\n")
                handle.write(truncated_tail)
                handle.flush()
                os.fsync(handle.fileno())

        manifest_lock = self._manifest_path.with_name(f"{self._manifest_path.name}.lock")
        with _exclusive_lock(manifest_lock):
            manifest: dict[str, object]
            if self._manifest_path.exists():
                with self._manifest_path.open(encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if not isinstance(loaded, dict):
                    raise ValueError("run_manifest.json must contain a JSON object")
                manifest = loaded
            else:
                manifest = {}
            repairs = manifest.setdefault("repairs", [])
            if not isinstance(repairs, list):
                raise ValueError("run_manifest.json repairs must be a JSON list")
            if repair not in repairs:
                repairs.append(repair)
                atomic_write_json(self._manifest_path, manifest)

        # The repair record is durable before the malformed bytes are excluded.
        _atomic_write_bytes(self.path, valid_bytes)

    def append_once(self, record: Mapping[str, object]) -> bool:
        canonical_record = json.loads(canonical_json(dict(record)))
        if not isinstance(canonical_record, dict):  # Defensive for type checkers and callers.
            raise ValueError("ledger record must be a JSON object")
        key = self._key_for(canonical_record)
        payload = canonical_json(canonical_record)

        with _exclusive_lock(self._lock_path):
            records = self._read_and_repair_locked()
            indexed = self._index_records(records)
            existing = indexed.get(key)
            if existing is not None:
                if existing != payload:
                    raise ValueError(f"conflicting payload for duplicate ledger key {key!r}")
                self._payload_by_key = indexed
                return False

            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                if self.path.stat().st_size:
                    with self.path.open("rb") as existing:
                        existing.seek(-1, os.SEEK_END)
                        if existing.read(1) not in (b"\n", b"\r"):
                            handle.write("\n")
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.path.parent)
            indexed[key] = payload
            self._payload_by_key = indexed
            return True

    def contains_key(self, key_values: Mapping[str, object]) -> bool:
        """Return whether the in-memory resume index has an exact ledger key."""
        if set(key_values) != set(self.key_fields):
            raise ValueError("ledger key values must match the configured key fields")
        key = tuple(key_values[field] for field in self.key_fields)
        return key in self._payload_by_key
