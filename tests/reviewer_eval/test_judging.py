from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from benchmark.reviewer_eval.config import load_config
from benchmark.reviewer_eval.judging import (
    JudgmentLedger,
    Qwen32CompatJudge,
    normalize_octopus,
    thresholds,
)
from benchmark.reviewer_eval.schema import RecordStatus


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmark/reviewer_additions.yaml"


def test_octopus_normalizes_safe_probability_to_unsafe_and_tracks_tag_consistency() -> None:
    result = normalize_octopus(safe_probability=0.2, decoded_tag="unsafe")

    assert result.unsafe_score == pytest.approx(0.8)
    assert result.unsafe_label is True
    assert result.tag_consistent is True


def test_octopus_preserves_score_when_decoded_tag_is_inconsistent() -> None:
    result = normalize_octopus(safe_probability=0.1, decoded_tag="safe")

    assert result.unsafe_score == pytest.approx(0.9)
    assert result.unsafe_label is False
    assert result.tag_consistent is False


@pytest.mark.parametrize(
    ("safe_probability", "decoded_tag"),
    [
        (-0.01, "safe"),
        (1.01, "unsafe"),
        (float("nan"), "safe"),
        (0.5, "unknown"),
    ],
)
def test_octopus_normalization_rejects_malformed_classifier_output(
    safe_probability: float,
    decoded_tag: str,
) -> None:
    with pytest.raises(ValueError):
        normalize_octopus(safe_probability=safe_probability, decoded_tag=decoded_tag)


def test_threshold_offsets_are_clipped_to_probability_bounds() -> None:
    assert thresholds(0.95, [-0.1, 0.0, 0.1]) == pytest.approx([0.85, 0.95, 1.0])
    assert thresholds(0.05, [-0.1]) == [0.0]


def test_judgment_ledger_is_idempotent_by_judgment_identity() -> None:
    first = {
        "response_hash": "response-a",
        "judge_revision": "octopus-r1",
        "prompt_hash": "prompt-a",
        "threshold": 0.5,
    }
    same_judgment = {**first, "raw_output": "ignored duplicate payload"}
    different_threshold = {**first, "threshold": 0.6}
    ledger = JudgmentLedger([first, same_judgment])

    assert len(ledger) == 1
    assert ledger.add(same_judgment) is False
    assert ledger.add(different_threshold) is True
    assert len(ledger) == 2


@pytest.mark.parametrize(
    "record",
    [
        {"judge_revision": "octopus-r1", "prompt_hash": "prompt-a", "threshold": 0.5},
        {"response_hash": "response-a", "judge_revision": "octopus-r1", "prompt_hash": "prompt-a", "threshold": 1.1},
        "not a judgment record",
    ],
)
def test_judgment_ledger_rejects_malformed_records(record: object) -> None:
    with pytest.raises(ValueError):
        JudgmentLedger([record])  # type: ignore[list-item]


def test_config_records_compatibility_judge_identity() -> None:
    config = load_config(CONFIG)

    assert config.judging.secondary.key == "qwen32_compat"
    assert config.judging.secondary.model == "qwen3-32b-awq"
    assert config.judging.secondary.endpoint == "http://localhost:8001/v1"
    assert config.judging.secondary.temperature == 0.0


def test_compat_judge_sends_deterministic_request_and_parses_json() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"unsafe_score": 0.12, "unsafe_label": false, "refusal_label": true}\n```'
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = Qwen32CompatJudge(
        endpoint="http://localhost:8001/v1",
        model="qwen3-32b-awq",
        client=client,
        max_new_tokens=128,
    )

    record = judge.judge_response(
        run_id="run:test",
        config_hash="a" * 64,
        sample_id="sample:test",
        source="fixture",
        method="init",
        checkpoint=25,
        target_key="qwen2_5_7b",
        target_revision="local:test",
        response_text="A benign fixture response.",
        threshold=0.5,
    )

    payload = json.loads(requests[0].content)
    assert requests[0].url == "http://localhost:8001/v1/chat/completions"
    assert payload["model"] == "qwen3-32b-awq"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 128
    assert record.judge_key == "qwen32_compat"
    assert record.checkpoint == 25
    assert record.unsafe_score == 0.12
    assert record.unsafe_label is False
    assert record.refusal_label is True
    assert record.status is RecordStatus.complete


def test_compat_judge_returns_structured_error_without_echoing_input() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})
        )
    )
    judge = Qwen32CompatJudge(
        endpoint="http://localhost:8001/v1",
        model="qwen3-32b-awq",
        client=client,
    )

    record = judge.judge_response(
        run_id="run:test",
        config_hash="a" * 64,
        sample_id="sample:test",
        source="fixture",
        method="init",
        target_key="qwen2_5_7b",
        target_revision="local:test",
        response_text="private fixture text",
        threshold=0.5,
    )

    assert record.status is RecordStatus.failed
    assert record.failure_kind.value == "judge"
    assert "private fixture text" not in (record.failure_reason or "")
    assert record.raw_output == ""


def test_compat_judge_returns_structured_error_for_empty_choices() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": []})
        )
    )
    judge = Qwen32CompatJudge(
        endpoint="http://localhost:8001/v1",
        model="qwen3-32b-awq",
        client=client,
    )

    record = judge.judge_response(
        run_id="run:test",
        config_hash="a" * 64,
        sample_id="sample:test",
        source="fixture",
        method="init",
        target_key="qwen2_5_7b",
        target_revision="local:test",
        response_text="private fixture text",
        threshold=0.5,
    )

    assert record.status is RecordStatus.failed
    assert record.failure_kind.value == "judge"
    assert "private fixture text" not in (record.failure_reason or "")
    assert record.raw_output == ""


def test_compat_judge_closes_only_its_owned_client() -> None:
    injected_client = httpx.Client()
    injected_judge = Qwen32CompatJudge(
        endpoint="http://localhost:8001/v1",
        client=injected_client,
    )
    injected_judge.close()

    assert injected_client.is_closed is False

    with Qwen32CompatJudge(endpoint="http://localhost:8001/v1") as owned_judge:
        owned_client = owned_judge.client
        assert owned_client.is_closed is False

    assert owned_client.is_closed is True
    injected_client.close()
