from __future__ import annotations

import json
from pathlib import Path

from benchmark.reviewer_eval.verification import verify_optimization_matrix


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_optimization_verification_requires_every_expected_checkpoint(tmp_path: Path) -> None:
    _write(
        tmp_path / "manifests" / "controlled_source.jsonl",
        [{"example_id": "source:001"}, {"example_id": "source:002"}],
    )
    _write(
        tmp_path / "optimization" / "source" / "init" / "records.jsonl",
        [
            {"sample_id": "source:001", "checkpoint": 0, "status": "complete"},
            {"sample_id": "source:002", "checkpoint": 0, "status": "complete"},
        ],
    )
    _write(
        tmp_path / "optimization" / "source" / "zol" / "records.jsonl",
        [
            {"sample_id": "source:001", "checkpoint": checkpoint, "status": "complete"}
            for checkpoint in (0, 25, 50, 100)
        ]
        + [
            {"sample_id": "source:002", "checkpoint": checkpoint, "status": "complete"}
            for checkpoint in (0, 25, 50)
        ],
    )

    result = verify_optimization_matrix(
        tmp_path,
        sources=("source",),
        methods=("init", "zol"),
    )

    assert result.complete is False
    assert result.expected_records == 10
    assert result.observed_records == 9
    assert result.errors == ("source/zol missing 1 terminal checkpoint",)
