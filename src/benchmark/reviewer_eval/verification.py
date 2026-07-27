"""Read-only completeness checks for the locked optimization matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .io import read_jsonl


_TERMINAL_STATUSES = frozenset(("complete", "failed"))


@dataclass(frozen=True)
class OptimizationVerification:
    complete: bool
    expected_records: int
    observed_records: int
    errors: tuple[str, ...]


def verify_optimization_matrix(
    output_root: str | Path,
    *,
    sources: Sequence[str],
    methods: Sequence[str],
) -> OptimizationVerification:
    """Verify every manifest sample has every required terminal checkpoint."""
    root = Path(output_root)
    expected_records = 0
    observed_records = 0
    errors: list[str] = []

    for source in sources:
        manifest_path = root / "manifests" / f"controlled_{source}.jsonl"
        sample_ids = {
            row["example_id"]
            for row in read_jsonl(manifest_path)
            if isinstance(row.get("example_id"), str)
        }
        if not sample_ids:
            errors.append(f"{source} missing controlled manifest samples")
            continue

        for method in methods:
            checkpoints = (0,) if method == "init" else (0, 25, 50, 100)
            expected = {
                (sample_id, checkpoint)
                for sample_id in sample_ids
                for checkpoint in checkpoints
            }
            expected_records += len(expected)
            records_path = root / "optimization" / source / method / "records.jsonl"
            observed = {
                (sample_id, checkpoint)
                for row in read_jsonl(records_path)
                if row.get("status") in _TERMINAL_STATUSES
                and isinstance((sample_id := row.get("sample_id")), str)
                and isinstance((checkpoint := row.get("checkpoint")), int)
                and (sample_id, checkpoint) in expected
            }
            observed_records += len(observed)
            missing = len(expected - observed)
            if missing:
                errors.append(f"{source}/{method} missing {missing} terminal checkpoint")

    return OptimizationVerification(
        complete=not errors,
        expected_records=expected_records,
        observed_records=observed_records,
        errors=tuple(errors),
    )
