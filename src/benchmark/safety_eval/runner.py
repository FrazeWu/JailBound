"""Transactional checkpoint persistence for tensor-only optimization runs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
from typing import Any, Literal

import torch

from .io import JsonlLedger, atomic_write_json, read_jsonl
from .schema import ComputeCounters, OptimizationRecord, RecordStatus


class RunConfigMismatch(RuntimeError):
    """Raised when an output root was initialized with a different config hash."""


class SerialBarrierError(RuntimeError):
    """Raised when a target is started before its predecessor is complete."""


@dataclass(frozen=True)
class SerialTargetBarrier:
    """Persist target completion only after responses and both judges terminate."""

    output_root: Path
    targets: tuple[str, ...]

    def __init__(self, output_root: str | Path, targets: tuple[str, ...]) -> None:
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("targets must be a non-empty unique serial order")
        object.__setattr__(self, "output_root", Path(output_root))
        object.__setattr__(self, "targets", targets)

    def require_ready(self, target: str) -> None:
        try:
            position = self.targets.index(target)
        except ValueError as error:
            raise SerialBarrierError(f"unknown target: {target}") from error
        if position and not self._marker_path(self.targets[position - 1]).is_file():
            raise SerialBarrierError(f"{self.targets[position - 1]} is incomplete")

    def mark_complete(
        self,
        target: str,
        *,
        response_count: int,
        primary_judgment_count: int,
        secondary_judgment_count: int,
    ) -> None:
        self.require_ready(target)
        counts = (response_count, primary_judgment_count, secondary_judgment_count)
        if any(count <= 0 for count in counts):
            raise SerialBarrierError("terminal counts must be positive for responses and both judges")
        if len(set(counts)) != 1:
            raise SerialBarrierError("terminal counts must agree across responses and both judges")
        atomic_write_json(
            self._marker_path(target),
            {"target": target, "terminal_count": response_count},
        )

    def _marker_path(self, target: str) -> Path:
        return self.output_root / "responses" / target / "TARGET_COMPLETE.json"


@dataclass(frozen=True)
class OptimizationJob:
    """Content-free identity for one source/method optimization trajectory."""

    source: str
    method: str
    cell_id: str
    sample_id: str
    random_seed: int


@dataclass(frozen=True)
class OptimizationSnapshot:
    """Minimal optimizer checkpoint payload accepted by the generic runner."""

    checkpoint: int
    representation: str
    attack_loss: float | None
    counters: ComputeCounters
    fol: float | None = None
    internal_margin: float | None = None
    state_filename: str | None = None
    state: dict[str, Any] | None = None


SnapshotFactory = Callable[[tuple[int, ...]], Iterable[OptimizationSnapshot]]
_TERMINAL_STATUSES = frozenset((RecordStatus.complete.value, RecordStatus.failed.value))


@dataclass
class OptimizationRunner:
    """Append exact checkpoint records and resume only missing terminal keys."""

    output_root: Path
    config_hash: str
    run_id: str
    git_revision: str
    schema_version: str = "reviewer_eval.v1"

    def __init__(
        self,
        output_root: str | Path,
        *,
        config_hash: str,
        run_id: str,
        git_revision: str,
        schema_version: str = "reviewer_eval.v1",
    ) -> None:
        self.output_root = Path(output_root)
        self.config_hash = config_hash
        self.run_id = run_id
        self.git_revision = git_revision
        self.schema_version = schema_version
        self._lock_config_hash()

    @property
    def _config_path(self) -> Path:
        return self.output_root / "optimization_config.json"

    @property
    def _config_lock_path(self) -> Path:
        return self.output_root / ".optimization_config.lock"

    def records_path(self, job: OptimizationJob) -> Path:
        self._validate_job(job)
        return self.output_root / "optimization" / job.source / job.method / "records.jsonl"

    def run(
        self,
        job: OptimizationJob,
        *,
        checkpoints: Iterable[int],
        snapshot_factory: SnapshotFactory,
    ) -> list[OptimizationRecord]:
        self._lock_config_hash()
        checkpoint_tuple = self._validate_checkpoints(checkpoints)
        path = self.records_path(job)
        ledger = JsonlLedger(path, key_fields=("cell_id", "sample_id", "checkpoint"))
        terminal = self._terminal_checkpoints(path, job)
        pending = tuple(checkpoint for checkpoint in checkpoint_tuple if checkpoint not in terminal)
        if not pending:
            return []

        written: list[OptimizationRecord] = []
        expected = set(pending)
        observed: set[int] = set()
        for snapshot in snapshot_factory(pending):
            if snapshot.checkpoint not in expected or snapshot.checkpoint in observed:
                raise ValueError("snapshot factory returned an unexpected checkpoint")
            observed.add(snapshot.checkpoint)
            record = self._record_for(job, snapshot, path.parent)
            if ledger.append_once(record.model_dump(mode="json")):
                written.append(record)
        if observed != expected:
            raise ValueError("snapshot factory did not return every requested checkpoint")
        return written

    def _lock_config_hash(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        with self._exclusive_config_lock():
            if self._config_path.exists():
                with self._config_path.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, dict) or payload.get("config_hash") != self.config_hash:
                    raise RunConfigMismatch("output root already has a different config hash")
                return
            atomic_write_json(self._config_path, {"config_hash": self.config_hash})

    @contextmanager
    def _exclusive_config_lock(self) -> Iterator[None]:
        with self._config_lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _validate_checkpoints(checkpoints: Iterable[int]) -> tuple[int, ...]:
        values = tuple(checkpoints)
        if not values or values != tuple(sorted(set(values))) or any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("checkpoints must be unique, ordered non-negative integers")
        return values

    @staticmethod
    def _validate_job(job: OptimizationJob) -> None:
        for value in (job.source, job.method):
            if not value or Path(value).name != value or value in {".", ".."}:
                raise ValueError("source and method must be single path components")

    @staticmethod
    def _terminal_checkpoints(path: Path, job: OptimizationJob) -> set[int]:
        terminal: set[int] = set()
        for row in read_jsonl(path):
            if (
                row.get("cell_id") == job.cell_id
                and row.get("sample_id") == job.sample_id
                and row.get("status") in _TERMINAL_STATUSES
                and isinstance(row.get("checkpoint"), int)
            ):
                terminal.add(row["checkpoint"])
        return terminal

    def _record_for(
        self,
        job: OptimizationJob,
        snapshot: OptimizationSnapshot,
        method_directory: Path,
    ) -> OptimizationRecord:
        state_path = self._persist_state(job, snapshot, method_directory)
        return OptimizationRecord(
            schema_version=self.schema_version,
            run_id=self.run_id,
            config_hash=self.config_hash,
            git_revision=self.git_revision,
            cell_id=job.cell_id,
            sample_id=job.sample_id,
            source=job.source,
            method=job.method,
            checkpoint=snapshot.checkpoint,
            random_seed=job.random_seed,
            status=RecordStatus.complete,
            failure_kind=None,
            failure_reason=None,
            state_path=state_path,
            representation=snapshot.representation,
            attack_loss=snapshot.attack_loss,
            fol=snapshot.fol,
            internal_margin=snapshot.internal_margin,
            materialized_prompt=None,
            counters=snapshot.counters,
        )

    @staticmethod
    def _state_path(state_filename: str | None, method_directory: Path) -> str | None:
        if state_filename is None:
            return None
        if not state_filename or Path(state_filename).name != state_filename:
            raise ValueError("state filename must be adjacent to records.jsonl")
        return str(method_directory / state_filename)

    def _persist_state(
        self,
        job: OptimizationJob,
        snapshot: OptimizationSnapshot,
        method_directory: Path,
    ) -> str | None:
        if snapshot.state is None:
            return self._state_path(snapshot.state_filename, method_directory)
        if snapshot.state_filename is not None:
            raise ValueError("state payload uses the runner-managed state path")
        state_id = stable_state_id(job, snapshot.checkpoint)
        state_path = method_directory / "states" / f"{state_id}.pt"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = state_path.with_suffix(".tmp")
        torch.save(snapshot.state, temporary_path)
        temporary_path.replace(state_path)
        return str(state_path)


def stable_state_id(job: OptimizationJob, checkpoint: int) -> str:
    """Return a content-free, collision-resistant checkpoint state filename stem."""
    import hashlib

    payload = f"{job.cell_id}|{job.sample_id}|{checkpoint}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]
