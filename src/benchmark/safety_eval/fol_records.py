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
CPU_OFFLOAD_RECOVERY_METHOD = f"{RECOVERY_METHOD}_cpu_offload"
TWO_GPU_RECOVERY_METHOD = f"{RECOVERY_METHOD}_two_gpu"
TWO_GPU_CHECKPOINTED_RECOVERY_METHOD = f"{RECOVERY_METHOD}_two_gpu_checkpointed"
FINITE_DIFFERENCE_RECOVERY_METHOD = f"{RECOVERY_METHOD}_fd"
FINITE_DIFFERENCE_SDPA_RECOVERY_METHOD = f"{FINITE_DIFFERENCE_RECOVERY_METHOD}_sdpa"
RECOVERY_METHODS = (
    RECOVERY_METHOD,
    EAGER_RECOVERY_METHOD,
    SDPA_RECOVERY_METHOD,
    EAGER_RETRY_METHOD,
    CHECKPOINTED_RECOVERY_METHOD,
    REBALANCED_RECOVERY_METHOD,
    CPU_OFFLOAD_RECOVERY_METHOD,
    TWO_GPU_RECOVERY_METHOD,
    TWO_GPU_CHECKPOINTED_RECOVERY_METHOD,
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
        if successful:
            # Recovery attempts are preserved in their own ledgers.  When a prior
            # eager attempt and a checkpointed retry both succeed, use eager: it
            # is the least-modified execution of the registered O+ path.
            priority = {EAGER_RECOVERY_METHOD: 0, CHECKPOINTED_RECOVERY_METHOD: 1}
            ranked = sorted(successful, key=lambda record: priority.get(str(record.get("method")), 99))
            if len(ranked) > 1 and priority.get(str(ranked[0].get("method")), 99) == priority.get(
                str(ranked[1].get("method")), 99
            ):
                raise ValueError("multiple equally preferred successful FOL recoveries for one failed primary")
            resolved[sample_id] = ranked[0]
    if recovery:
        raise ValueError("recovery references an unknown primary FOL record")
    return resolved
