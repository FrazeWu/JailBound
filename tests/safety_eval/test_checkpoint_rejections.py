from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.safety_eval.checkpoint_rejections import (
    ManualCheckpointRejection,
    load_manual_checkpoint_rejections,
)


def _write_jsonl(path: Path, *rows: object) -> None:
    path.write_text(
        "".join(
            row if isinstance(row, str) else json.dumps(row, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_load_manual_checkpoint_rejections_accepts_strict_jsonl(tmp_path: Path) -> None:
    ledger = tmp_path / "manual_rejections.jsonl"
    _write_jsonl(
        ledger,
        {
            "branch": "jailbound_o_plus",
            "step": 225,
            "state_sha256": "a" * 64,
            "reason": "Readable ASCII but code-like English",
        },
        {
            "branch": "jailbound_o_minus",
            "step": 250,
            "state_sha256": "b" * 64,
            "reason": "Retains harmful structure",
        },
    )

    loaded = load_manual_checkpoint_rejections(ledger)

    assert loaded == (
        ManualCheckpointRejection(
            branch="jailbound_o_plus",
            step=225,
            state_sha256="a" * 64,
            reason="Readable ASCII but code-like English",
        ),
        ManualCheckpointRejection(
            branch="jailbound_o_minus",
            step=250,
            state_sha256="b" * 64,
            reason="Retains harmful structure",
        ),
    )
    assert loaded[0].branch_step == ("jailbound_o_plus", 225)
    assert loaded[0].evidence() == {
        "branch": "jailbound_o_plus",
        "step": 225,
        "state_sha256": "a" * 64,
        "reason": "Readable ASCII but code-like English",
    }
    assert load_manual_checkpoint_rejections(None) == ()


def test_load_manual_checkpoint_rejections_accepts_exact_fields_in_different_key_order(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "manual_rejections.jsonl"
    _write_jsonl(
        ledger,
        {
            "step": 250,
            "branch": "jailbound_o_minus",
            "reason": "Retains harmful structure",
            "state_sha256": "b" * 64,
        },
        {
            "branch": "jailbound_o_plus",
            "step": 225,
            "state_sha256": "a" * 64,
            "reason": "Readable ASCII but code-like English",
        },
    )

    loaded = load_manual_checkpoint_rejections(ledger)

    assert loaded == (
        ManualCheckpointRejection(
            branch="jailbound_o_minus",
            step=250,
            state_sha256="b" * 64,
            reason="Retains harmful structure",
        ),
        ManualCheckpointRejection(
            branch="jailbound_o_plus",
            step=225,
            state_sha256="a" * 64,
            reason="Readable ASCII but code-like English",
        ),
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (['{"branch":"jailbound_o_plus","step":225,"state_sha256":"a"'], "line 1"),
        ([{
            "branch": "jailbound_o_plus",
            "step": 225,
            "state_sha256": "a" * 64,
            "reason": "Readable ASCII but code-like English",
            "extra": True,
        }], "exactly four fields"),
        ([{
            "branch": "unknown",
            "step": 225,
            "state_sha256": "a" * 64,
            "reason": "Readable ASCII but code-like English",
        }], "unknown branch"),
        ([{
            "branch": "jailbound_o_plus",
            "step": 0,
            "state_sha256": "a" * 64,
            "reason": "Readable ASCII but code-like English",
        }], "positive integer"),
        ([{
            "branch": "jailbound_o_plus",
            "step": True,
            "state_sha256": "a" * 64,
            "reason": "Readable ASCII but code-like English",
        }], "positive integer"),
        ([{
            "branch": "jailbound_o_plus",
            "step": 225,
            "state_sha256": "A" * 64,
            "reason": "Readable ASCII but code-like English",
        }], "lowercase SHA-256"),
        ([{
            "branch": "jailbound_o_plus",
            "step": 225,
            "state_sha256": "a" * 63,
            "reason": "Readable ASCII but code-like English",
        }], "lowercase SHA-256"),
        ([{
            "branch": "jailbound_o_plus",
            "step": 225,
            "state_sha256": "a" * 64,
            "reason": "   ",
        }], "non-blank reason"),
    ],
)
def test_load_manual_checkpoint_rejections_rejects_invalid_rows(
    tmp_path: Path,
    rows: list[object],
    message: str,
) -> None:
    ledger = tmp_path / "manual_rejections.jsonl"
    _write_jsonl(ledger, *rows)

    with pytest.raises(ValueError, match=message):
        load_manual_checkpoint_rejections(ledger)


def test_load_manual_checkpoint_rejections_rejects_duplicate_identity(tmp_path: Path) -> None:
    ledger = tmp_path / "manual_rejections.jsonl"
    row = {
        "branch": "jailbound_o_plus",
        "step": 225,
        "state_sha256": "a" * 64,
        "reason": "Readable ASCII but code-like English",
    }
    _write_jsonl(ledger, row, row)

    with pytest.raises(ValueError, match="duplicate identity"):
        load_manual_checkpoint_rejections(ledger)


def test_load_manual_checkpoint_rejections_rejects_duplicate_branch_step_with_different_hash(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "manual_rejections.jsonl"
    _write_jsonl(
        ledger,
        {
            "branch": "jailbound_o_plus",
            "step": 225,
            "state_sha256": "a" * 64,
            "reason": "Readable ASCII but code-like English",
        },
        {
            "branch": "jailbound_o_plus",
            "step": 225,
            "state_sha256": "b" * 64,
            "reason": "Same step, different state",
        },
    )

    with pytest.raises(ValueError, match="duplicate branch/step"):
        load_manual_checkpoint_rejections(ledger)
