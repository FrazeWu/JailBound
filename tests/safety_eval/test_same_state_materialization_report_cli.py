from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from benchmark.safety_eval.materialization_ablation import MaterializationPair


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/report_same_state_materialization_ablation.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("same_state_materialization_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair(
    sample_id: str,
    *,
    branch: str = "High-Value",
    state_sha256: str = "a" * 64,
    transport: str = "direct_projected_token_ids",
) -> MaterializationPair:
    return MaterializationPair.model_validate({
        "source": "harmbench",
        "sample_id": sample_id,
        "branch": branch,
        "optimization_checkpoint": 100,
        "state_sha256": state_sha256,
        "model_key": "qwen2_5_7b",
        "model_revision": "local",
        "initial_discrete_prompt": "initial",
        "reference_intent": "intent",
        "continuous_response": "continuous",
        "materialized_text": "materialized",
        "materialized_response": "materialized response",
        "materialized_transport": transport,
        "editable_projected_token_ids": [1, 2],
        "projected_token_ids": [1, 3, 2],
        "retokenized_token_ids": [1, 3, 2],
        "projection_cosines": [0.9, 0.8],
        "roundtrip_exact_match": True,
        "projected_length": 3,
        "retokenized_length": 3,
        "max_new_tokens": 512,
        "status": "complete",
        "error": None,
        "judgments": {},
    })


def _judgment(
    sample_id: str,
    *,
    continuous: bool,
    materialized: bool,
    branch: str = "High-Value",
) -> dict[str, object]:
    return {
        "source": "harmbench",
        "sample_id": sample_id,
        "branch": branch,
        "optimization_checkpoint": 100,
        "run_id": "materialization:fixture",
        "config_hash": "b" * 64,
        "target_key": "qwen2_5_7b",
        "target_revision": "local",
        "judge_key": "octopus_seval_14b",
        "judge_revision": "fixture-r1",
        "threshold": 0.5,
        "continuous_score": 0.9 if continuous else 0.1,
        "materialized_score": 0.9 if materialized else 0.1,
        "continuous": continuous,
        "materialized": materialized,
        "continuous_raw_output": "unsafe" if continuous else "safe",
        "materialized_raw_output": "unsafe" if materialized else "safe",
        "status": "complete",
        "error": None,
    }


def test_build_report_emits_requested_asr_columns_and_difference_direction() -> None:
    module = _load_script()
    pairs = (_pair("a"), _pair("b"))
    judgments = (
        module.PairedJudgmentRecord.model_validate(
            _judgment("a", continuous=True, materialized=False)
        ),
        module.PairedJudgmentRecord.model_validate(
            _judgment("b", continuous=False, materialized=True)
        ),
    )

    report = module.build_report(pairs, judgments, judge_key="octopus_seval_14b")

    assert "| Source | Branch | Continuous ASR | Materialized ASR | Difference |" in report
    assert "| harmbench | High-Value | 50.0% | 50.0% | +0.0 pp |" in report
    assert "Difference = Materialized ASR - Continuous ASR" in report


def test_build_report_marks_cross_branch_projection_collapse() -> None:
    module = _load_script()
    pairs = (
        _pair("same", branch="High-Value", state_sha256="a" * 64),
        _pair("same", branch="Safety-Sensitivity", state_sha256="b" * 64),
    )
    judgments = (
        module.PairedJudgmentRecord.model_validate(
            _judgment("same", branch="High-Value", continuous=True, materialized=True)
        ),
        module.PairedJudgmentRecord.model_validate(
            _judgment(
                "same",
                branch="Safety-Sensitivity",
                continuous=False,
                materialized=True,
            )
        ),
    )

    report = module.build_report(pairs, judgments, judge_key="octopus_seval_14b")

    assert "Cross-branch projection collapse" in report
    assert "harmbench: 1/1 matched samples" in report
    assert "shared projected result" in report


def test_build_report_rejects_non_direct_materialized_transport() -> None:
    module = _load_script()
    pair = _pair("a", transport="decoded_retokenized_text")
    judgment = module.PairedJudgmentRecord.model_validate(
        _judgment("a", continuous=True, materialized=False)
    )

    with pytest.raises(ValueError, match="direct projected-token transport"):
        module.build_report((pair,), (judgment,), judge_key="octopus_seval_14b")


def test_build_report_rejects_incomplete_primary_judgment_coverage() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="coverage"):
        module.build_report((_pair("a"),), (), judge_key="octopus_seval_14b")
