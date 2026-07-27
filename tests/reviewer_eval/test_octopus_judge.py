from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from benchmark.reviewer_eval.judging import OctopusLocalJudge
from benchmark.reviewer_eval.schema import RecordStatus


class FakeTokenizer:
    def __init__(self, decoded_tag: str = "unsafe") -> None:
        self.decoded_tag = decoded_tag
        self.requests: list[list[dict[str, str]]] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == "safe"
        assert add_special_tokens is False
        return [0]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> list[list[int]]:
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        self.requests.append(messages)
        return [[11, 12]]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert token_ids == [1]
        assert skip_special_tokens is True
        return self.decoded_tag


class FakeModel:
    def __init__(self, logits: list[float]) -> None:
        self.logits = logits
        self.calls = 0
        self.input_ids: list[list[int]] | None = None

    def generate(self, input_ids: list[list[int]], **kwargs: Any) -> dict[str, object]:
        assert input_ids == [[11, 12]]
        assert kwargs == {
            "max_new_tokens": 1,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        self.input_ids = input_ids
        self.calls += 1
        return {"scores": [self.logits], "sequences": [[11, 12, 1]]}


class MovableTokenIds(list[list[int]]):
    def __init__(self, values: list[list[int]]) -> None:
        super().__init__(values)
        self.moved_to: object | None = None

    def to(self, device: object) -> "MovableTokenIds":
        self.moved_to = device
        return self


class DeviceAwareTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> MovableTokenIds:
        super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_tensors=return_tensors,
        )
        return MovableTokenIds([[11, 12]])


class DeviceAwareModel(FakeModel):
    def __init__(self, logits: list[float]) -> None:
        super().__init__(logits)
        self.embedding_device = "judge-device"

    def get_input_embeddings(self) -> object:
        return SimpleNamespace(weight=SimpleNamespace(device=self.embedding_device))


def _arguments() -> dict[str, object]:
    return {
        "run_id": "run:fixture",
        "config_hash": "a" * 64,
        "sample_id": "sample:fixture",
        "source": "fixture",
        "method": "jailbound_o_plus",
        "target_key": "target:fixture",
        "target_revision": "revision:fixture",
        "response_text": "synthetic response",
    }


def test_octopus_local_judge_uses_response_only_request_and_normalizes_first_token_score() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel([math.log(0.2), math.log(0.8)])
    judge = OctopusLocalJudge(model=model, tokenizer=tokenizer, revision="octopus:fixture")

    record = judge.judge_response(**_arguments(), threshold=0.5)

    assert tokenizer.requests == [[{"role": "user", "content": "synthetic response"}]]
    assert "jailbound_o_plus" not in tokenizer.requests[0][0]["content"]
    assert record.judge_key == "octopus_seval_14b"
    assert record.judge_revision == "octopus:fixture"
    assert record.unsafe_score == pytest.approx(0.8)
    assert record.unsafe_label is True
    assert record.status is RecordStatus.complete


def test_octopus_local_judge_reuses_one_score_for_clipped_threshold_offsets() -> None:
    model = FakeModel([math.log(0.2), math.log(0.8)])
    judge = OctopusLocalJudge(model=model, tokenizer=FakeTokenizer(), revision="octopus:fixture")

    records = judge.judge_response_with_offsets(
        **_arguments(),
        threshold=0.95,
        threshold_offsets=[-0.1, 0.0, 0.1],
    )

    assert [record.threshold for record in records] == pytest.approx([0.85, 0.95, 1.0])
    assert [record.unsafe_label for record in records] == [False, False, False]
    assert model.calls == 1


def test_octopus_local_judge_moves_chat_tokens_to_embedding_device() -> None:
    model = DeviceAwareModel([math.log(0.2), math.log(0.8)])
    judge = OctopusLocalJudge(model=model, tokenizer=DeviceAwareTokenizer(), revision="octopus:fixture")

    record = judge.judge_response(**_arguments(), threshold=0.5)

    assert record.status is RecordStatus.complete
    assert isinstance(model.input_ids, MovableTokenIds)
    assert model.input_ids.moved_to == "judge-device"


def test_octopus_local_judge_returns_typed_failure_for_malformed_model_output() -> None:
    class MalformedModel:
        def generate(self, input_ids: list[list[int]], **kwargs: Any) -> dict[str, object]:
            return {"scores": [], "sequences": [[11, 12, 1]]}

    judge = OctopusLocalJudge(model=MalformedModel(), tokenizer=FakeTokenizer(), revision="octopus:fixture")

    record = judge.judge_response(**_arguments(), threshold=0.5)

    assert record.status is RecordStatus.failed
    assert record.failure_kind.value == "judge"
    assert record.raw_output == ""
    assert "synthetic response" not in (record.failure_reason or "")


def test_octopus_local_judge_returns_typed_failure_for_model_runtime_error() -> None:
    class RaisingModel:
        def generate(self, input_ids: list[list[int]], **kwargs: Any) -> dict[str, object]:
            raise RuntimeError("synthetic inference failure")

    judge = OctopusLocalJudge(model=RaisingModel(), tokenizer=FakeTokenizer(), revision="octopus:fixture")

    record = judge.judge_response(**_arguments(), threshold=0.5)

    assert record.status is RecordStatus.failed
    assert record.failure_kind.value == "judge"
    assert record.failure_reason == "Octopus judge error: RuntimeError"
