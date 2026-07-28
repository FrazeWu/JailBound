"""Resolve FOL terminal checkpoints while preserving failed primary attempts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .io import read_jsonl


PRIMARY_METHOD = "jailbound_o_plus"
RECOVERY_METHOD = f"{PRIMARY_METHOD}_recovery"
EAGER_RECOVERY_METHOD = f"{RECOVERY_METHOD}_eager"
SDPA_RECOVERY_METHOD = f"{RECOVERY_METHOD}_sdpa"
EAGER_RETRY_METHOD = f"{RECOVERY_METHOD}_eager_retry"
CHECKPOINTED_RECOVERY_METHOD = f"{RECOVERY_METHOD}_checkpointed"
REBALANCED_RECOVERY_METHOD = f"{RECOVERY_METHOD}_rebalanced"
FINITE_DIFFERENCE_RECOVERY_METHOD = f"{RECOVERY_METHOD}_fd"
FINITE_DIFFERENCE_SDPA_RECOVERY_METHOD = f"{FINITE_DIFFERENCE_RECOVERY_METHOD}_sdpa"
RECOVERY_METHODS = (
    RECOVERY_METHOD,
    EAGER_RECOVERY_METHOD,
    SDPA_RECOVERY_METHOD,
    EAGER_RETRY_METHOD,
    CHECKPOINTED_RECOVERY_METHOD,
    REBALANCED_RECOVERY_METHOD,
    FINITE_DIFFERENCE_RECOVERY_METHOD,
    FINITE_DIFFERENCE_SDPA_RECOVERY_METHOD,
)


def _terminal_by_sample(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for row in read_jsonl(path):
        if row.get("checkpoint") != 100 or not isinstance(row.get("sample_id"), str):
            continue
        sample_id = str(row["sample_id"])
        if sample_id in records:
            raise ValueError("duplicate FOL terminal record")
        records[sample_id] = row
    return records


def resolved_terminal_payloads(root: str | Path, source: str) -> dict[str, dict[str, object]]:
    """Return primary completions plus one successful recovery for failed primaries."""

    optimization = Path(root) / "optimization" / source
    primary = _terminal_by_sample(optimization / PRIMARY_METHOD / "records.jsonl")
    recovery: dict[str, list[dict[str, object]]] = {}
    for method in RECOVERY_METHODS:
        for sample_id, record in _terminal_by_sample(optimization / method / "records.jsonl").items():
            recovery.setdefault(sample_id, []).append(record)
    resolved: dict[str, dict[str, object]] = {}
    for sample_id, primary_record in primary.items():
        attempts = recovery.pop(sample_id, [])
        if primary_record.get("status") == "complete":
            if attempts:
                raise ValueError("recovery exists for a completed primary FOL record")
            resolved[sample_id] = primary_record
            continue
        successful = [record for record in attempts if record.get("status") == "complete"]
        if len(successful) > 1:
            raise ValueError("multiple successful FOL recoveries for one failed primary")
        if successful:
            resolved[sample_id] = successful[0]
    if recovery:
        raise ValueError("recovery references an unknown primary FOL record")
    return resolved
