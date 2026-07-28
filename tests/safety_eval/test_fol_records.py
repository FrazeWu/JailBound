from __future__ import annotations

from benchmark.safety_eval.fol_records import resolved_terminal_payloads
from benchmark.safety_eval.io import atomic_write_jsonl


def _record(sample_id: str, *, status: str, method: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "checkpoint": 100,
        "status": status,
        "method": method,
    }


def test_resolved_terminal_payloads_use_recovery_only_for_failed_primary(tmp_path) -> None:
    root = tmp_path
    primary = root / "optimization" / "fixture" / "jailbound_o_plus" / "records.jsonl"
    recovery = root / "optimization" / "fixture" / "jailbound_o_plus_recovery" / "records.jsonl"
    atomic_write_jsonl(
        primary,
        (
            _record("sample:complete", status="complete", method="jailbound_o_plus"),
            _record("sample:failed", status="failed", method="jailbound_o_plus"),
        ),
    )
    atomic_write_jsonl(
        recovery,
        (_record("sample:failed", status="complete", method="jailbound_o_plus_recovery"),),
    )

    resolved = resolved_terminal_payloads(root, "fixture")

    assert set(resolved) == {"sample:complete", "sample:failed"}
    assert resolved["sample:complete"]["method"] == "jailbound_o_plus"
    assert resolved["sample:failed"]["method"] == "jailbound_o_plus_recovery"


def test_resolved_terminal_payloads_accepts_later_eager_recovery_after_failed_sdpa(tmp_path) -> None:
    primary = tmp_path / "optimization" / "fixture" / "jailbound_o_plus" / "records.jsonl"
    sdpa = tmp_path / "optimization" / "fixture" / "jailbound_o_plus_recovery" / "records.jsonl"
    eager = tmp_path / "optimization" / "fixture" / "jailbound_o_plus_recovery_eager" / "records.jsonl"
    atomic_write_jsonl(
        primary,
        (_record("sample:failed", status="failed", method="jailbound_o_plus"),),
    )
    atomic_write_jsonl(
        sdpa,
        (_record("sample:failed", status="failed", method="jailbound_o_plus_recovery"),),
    )
    atomic_write_jsonl(
        eager,
        (_record("sample:failed", status="complete", method="jailbound_o_plus_recovery_eager"),),
    )

    resolved = resolved_terminal_payloads(tmp_path, "fixture")

    assert set(resolved) == {"sample:failed"}
    assert resolved["sample:failed"]["method"] == "jailbound_o_plus_recovery_eager"


def test_resolved_terminal_payloads_accepts_sdpa_retry_after_failed_recoveries(tmp_path) -> None:
    primary = tmp_path / "optimization" / "fixture" / "jailbound_o_plus" / "records.jsonl"
    eager = tmp_path / "optimization" / "fixture" / "jailbound_o_plus_recovery_eager" / "records.jsonl"
    sdpa_retry = tmp_path / "optimization" / "fixture" / "jailbound_o_plus_recovery_sdpa" / "records.jsonl"
    atomic_write_jsonl(
        primary,
        (_record("sample:failed", status="failed", method="jailbound_o_plus"),),
    )
    atomic_write_jsonl(
        eager,
        (_record("sample:failed", status="failed", method="jailbound_o_plus_recovery_eager"),),
    )
    atomic_write_jsonl(
        sdpa_retry,
        (_record("sample:failed", status="complete", method="jailbound_o_plus_recovery_sdpa"),),
    )

    resolved = resolved_terminal_payloads(tmp_path, "fixture")

    assert resolved["sample:failed"]["method"] == "jailbound_o_plus_recovery_sdpa"


def test_resolved_terminal_payloads_accepts_eager_retry_after_failed_sdpa(tmp_path) -> None:
    primary = tmp_path / "optimization" / "fixture" / "jailbound_o_plus" / "records.jsonl"
    sdpa = tmp_path / "optimization" / "fixture" / "jailbound_o_plus_recovery_sdpa" / "records.jsonl"
    eager = tmp_path / "optimization" / "fixture" / "jailbound_o_plus_recovery_eager_retry" / "records.jsonl"
    atomic_write_jsonl(primary, (_record("sample:failed", status="failed", method="jailbound_o_plus"),))
    atomic_write_jsonl(sdpa, (_record("sample:failed", status="failed", method="jailbound_o_plus_recovery_sdpa"),))
    atomic_write_jsonl(eager, (_record("sample:failed", status="complete", method="jailbound_o_plus_recovery_eager_retry"),))

    resolved = resolved_terminal_payloads(tmp_path, "fixture")

    assert resolved["sample:failed"]["method"] == "jailbound_o_plus_recovery_eager_retry"
