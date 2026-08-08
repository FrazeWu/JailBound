from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from benchmark.safety_eval.materialization_ablation import MaterializationPair


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/analyze_materialization_ablation.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyze_materialization_ablation_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair(sample_id: str = "sample:1", *, status: str = "complete") -> MaterializationPair:
    failed = status == "failed"
    return MaterializationPair.model_validate({
        "source": "harmbench",
        "sample_id": sample_id,
        "branch": "High-Value",
        "optimization_checkpoint": 100,
        "state_sha256": None if failed else "a" * 64,
        "model_key": "qwen2_5_7b",
        "model_revision": "local",
        "initial_discrete_prompt": "initial prompt",
        "reference_intent": "reference intent",
        "continuous_response": "" if failed else "continuous response",
        "materialized_text": "" if failed else "materialized prompt",
        "materialized_response": "" if failed else "materialized response",
        "editable_projected_token_ids": [] if failed else [4],
        "projected_token_ids": [] if failed else [4],
        "retokenized_token_ids": [] if failed else [4],
        "projection_cosines": [] if failed else [0.9],
        "roundtrip_exact_match": not failed,
        "projected_length": 0 if failed else 1,
        "retokenized_length": 0 if failed else 1,
        "max_new_tokens": 512,
        "status": status,
        "error": "generation failed" if failed else None,
        "judgments": {},
    })


def _judgment(sample_id: str = "sample:1", **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "harmbench",
        "sample_id": sample_id,
        "branch": "High-Value",
        "optimization_checkpoint": 100,
        "run_id": "materialization:test",
        "config_hash": "b" * 64,
        "target_key": "qwen2_5_7b",
        "target_revision": "local",
        "judge_key": "octopus_seval_14b",
        "judge_revision": "octopus-fixture",
        "threshold": 0.5,
        "continuous_score": 0.8,
        "materialized_score": 0.2,
        "continuous": True,
        "materialized": False,
        "continuous_raw_output": "unsafe",
        "materialized_raw_output": "safe",
        "status": "complete",
        "error": None,
    }
    row.update(updates)
    return row


def _both_judgments(sample_id: str = "sample:1") -> list[dict[str, object]]:
    return [
        _judgment(sample_id),
        _judgment(
            sample_id,
            judge_key="qwen32_compat",
            judge_revision="endpoint:http://localhost:8001/v1;model:qwen3-32b-awq;temperature:0",
        ),
    ]


def _write_pairs(path: Path, pairs: list[MaterializationPair]) -> None:
    path.write_text("".join(pair.model_dump_json() + "\n" for pair in pairs), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_join_judgments_rejects_duplicate_row(tmp_path: Path) -> None:
    module = _load_script()
    pairs_path, judgments_path = tmp_path / "pairs.jsonl", tmp_path / "judgments.jsonl"
    _write_pairs(pairs_path, [_pair()])
    _write_rows(judgments_path, [_judgment(), _judgment()])

    with pytest.raises(ValueError, match="duplicate paired judgment"):
        module.join_judgments(pairs_path, judgments_path, tmp_path / "joined.jsonl")


def test_join_judgments_rejects_unknown_pair(tmp_path: Path) -> None:
    module = _load_script()
    pairs_path, judgments_path = tmp_path / "pairs.jsonl", tmp_path / "judgments.jsonl"
    _write_pairs(pairs_path, [_pair()])
    _write_rows(judgments_path, [_judgment("unknown")])

    with pytest.raises(ValueError, match="unknown pair"):
        module.join_judgments(pairs_path, judgments_path, tmp_path / "joined.jsonl")


def test_join_judgments_rejects_judgment_on_failed_pair(tmp_path: Path) -> None:
    module = _load_script()
    pairs_path, judgments_path = tmp_path / "pairs.jsonl", tmp_path / "judgments.jsonl"
    _write_pairs(pairs_path, [_pair(status="failed")])
    _write_rows(judgments_path, [_judgment()])

    with pytest.raises(ValueError, match="failed pair"):
        module.join_judgments(pairs_path, judgments_path, tmp_path / "joined.jsonl")


def test_join_judgments_rejects_incomplete_judge_coverage(tmp_path: Path) -> None:
    module = _load_script()
    pairs_path, judgments_path = tmp_path / "pairs.jsonl", tmp_path / "judgments.jsonl"
    _write_pairs(pairs_path, [_pair("sample:1"), _pair("sample:2")])
    _write_rows(judgments_path, _both_judgments("sample:1"))

    with pytest.raises(ValueError, match="incomplete"):
        module.join_judgments(pairs_path, judgments_path, tmp_path / "joined.jsonl")


def test_join_judgments_requires_both_frozen_judges(tmp_path: Path) -> None:
    module = _load_script()
    pairs_path, judgments_path = tmp_path / "pairs.jsonl", tmp_path / "judgments.jsonl"
    _write_pairs(pairs_path, [_pair()])
    _write_rows(judgments_path, [_judgment()])

    with pytest.raises(ValueError, match="expected judges"):
        module.join_judgments(pairs_path, judgments_path, tmp_path / "joined.jsonl")


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"judge_revision": ""}, "judge_revision"),
        ({"threshold": 0.4}, "threshold"),
        ({"status": "failed", "error": "judge failed", "continuous_score": None,
          "materialized_score": None, "continuous": None, "materialized": None}, "terminal"),
    ],
)
def test_join_judgments_rejects_unfrozen_or_failed_judgment(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    module = _load_script()
    pairs_path, judgments_path = tmp_path / "pairs.jsonl", tmp_path / "judgments.jsonl"
    _write_pairs(pairs_path, [_pair()])
    rows = _both_judgments()
    rows[0].update(updates)
    _write_rows(judgments_path, rows)

    with pytest.raises(ValueError, match=message):
        module.join_judgments(pairs_path, judgments_path, tmp_path / "joined.jsonl")


def test_join_judgments_rejects_empty_judgments(tmp_path: Path) -> None:
    module = _load_script()
    pairs_path, judgments_path = tmp_path / "pairs.jsonl", tmp_path / "judgments.jsonl"
    _write_pairs(pairs_path, [_pair()])
    judgments_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="judgments are empty"):
        module.join_judgments(pairs_path, judgments_path, tmp_path / "joined.jsonl")


def test_join_judgments_rejects_pairs_with_preexisting_labels(tmp_path: Path) -> None:
    module = _load_script()
    pairs_path, judgments_path = tmp_path / "pairs.jsonl", tmp_path / "judgments.jsonl"
    pair = _pair()
    payload = pair.model_dump(mode="json")
    payload["judgments"] = {"stale": {"continuous": True, "materialized": True}}
    _write_pairs(pairs_path, [MaterializationPair.model_validate(payload)])
    _write_rows(judgments_path, _both_judgments())

    with pytest.raises(ValueError, match="pre-existing judgments"):
        module.join_judgments(pairs_path, judgments_path, tmp_path / "joined.jsonl")


def test_join_judgments_rejects_empty_judge_key(tmp_path: Path) -> None:
    module = _load_script()
    pairs_path, judgments_path = tmp_path / "pairs.jsonl", tmp_path / "judgments.jsonl"
    _write_pairs(pairs_path, [_pair()])
    _write_rows(judgments_path, [_judgment(judge_key="")])

    with pytest.raises(ValueError, match="judge key must be non-empty"):
        module.join_judgments(pairs_path, judgments_path, tmp_path / "joined.jsonl")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_key", "different-target", "target provenance"),
        ("target_revision", "different-revision", "target provenance"),
        ("config_hash", "c" * 64, "config_hash"),
        ("run_id", "materialization:different", "run_id"),
    ],
)
def test_join_judgments_rejects_mixed_pair_or_run_provenance(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    module = _load_script()
    pairs_path, judgments_path = tmp_path / "pairs.jsonl", tmp_path / "judgments.jsonl"
    _write_pairs(pairs_path, [_pair()])
    rows = _both_judgments()
    rows[1][field] = value
    _write_rows(judgments_path, rows)

    with pytest.raises(ValueError, match=message):
        module.join_judgments(pairs_path, judgments_path, tmp_path / "joined.jsonl")


def test_join_judgments_rejects_multiple_revisions_for_one_judge(tmp_path: Path) -> None:
    module = _load_script()
    pairs_path, judgments_path = tmp_path / "pairs.jsonl", tmp_path / "judgments.jsonl"
    _write_pairs(pairs_path, [_pair("sample:1"), _pair("sample:2")])
    rows = _both_judgments("sample:1") + _both_judgments("sample:2")
    rows[2]["judge_revision"] = "different-octopus-revision"
    _write_rows(judgments_path, rows)

    with pytest.raises(ValueError, match="multiple revisions"):
        module.join_judgments(pairs_path, judgments_path, tmp_path / "joined.jsonl")


@pytest.mark.parametrize(
    "updates",
    [
        {"branch": "not-a-branch"},
        {"optimization_checkpoint": "100"},
        {"continuous": "true"},
        {"materialized": 0},
    ],
)
def test_join_judgments_rejects_malformed_identity_or_labels(tmp_path: Path, updates: dict[str, object]) -> None:
    module = _load_script()
    pairs_path, judgments_path = tmp_path / "pairs.jsonl", tmp_path / "judgments.jsonl"
    _write_pairs(pairs_path, [_pair()])
    _write_rows(judgments_path, [_judgment(**updates)])

    with pytest.raises(ValueError, match="invalid paired judgment at line 1"):
        module.join_judgments(pairs_path, judgments_path, tmp_path / "joined.jsonl")


def test_join_judgments_exactly_joins_complete_pairs(tmp_path: Path) -> None:
    module = _load_script()
    pairs_path, judgments_path, destination = (
        tmp_path / "pairs.jsonl",
        tmp_path / "judgments.jsonl",
        tmp_path / "joined.jsonl",
    )
    _write_pairs(pairs_path, [_pair("sample:2"), _pair("sample:1"), _pair("failed", status="failed")])
    _write_rows(judgments_path, _both_judgments("sample:1") + _both_judgments("sample:2"))

    judge_keys = module.join_judgments(pairs_path, judgments_path, destination)

    joined = [MaterializationPair.model_validate_json(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert judge_keys == ("octopus_seval_14b", "qwen32_compat")
    assert [pair.sample_id for pair in joined] == ["sample:2", "sample:1", "failed"]
    assert joined[0].judgments["octopus_seval_14b"].continuous is True
    assert joined[0].judgments["octopus_seval_14b"].materialized is False
    assert joined[1].judgments == joined[0].judgments
    assert joined[2].judgments == {}
