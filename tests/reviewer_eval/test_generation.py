from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from benchmark.reviewer_eval.generation import generate_one, generate_response_record, shard_ids
from benchmark.reviewer_eval.schema import MaterializationRecord, RecordStatus


class FakeTokenizer:
    def __init__(self, *, rejects_system_role: bool = False) -> None:
        self.rejects_system_role = rejects_system_role
        self.messages: list[list[dict[str, str]]] = []
        self.decoded_tokens: list[int] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> list[list[int]]:
        if self.rejects_system_role and any(message["role"] == "system" for message in messages):
            raise ValueError("system role is unsupported")
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        self.messages.append(messages)
        return [[11, 12, 13]]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decoded_tokens = token_ids
        return "response:" + ",".join(str(token_id) for token_id in token_ids)


class FakeModel:
    def __init__(self) -> None:
        self.input_ids: list[list[int]] | None = None
        self.generate_kwargs: dict[str, object] | None = None

    def generate(self, input_ids: list[list[int]], **kwargs: object) -> list[list[int]]:
        self.input_ids = input_ids
        self.generate_kwargs = kwargs
        return [input_ids[0] + [91, 92]]


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
        return MovableTokenIds([[11, 12, 13]])


class DeviceAwareModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_device = "target-device"

    def get_input_embeddings(self) -> object:
        return SimpleNamespace(weight=SimpleNamespace(device=self.embedding_device))


def test_shard_ids_is_deterministic_and_covers_each_input_once() -> None:
    ids = ("example-c", "example-a", "example-c", "example-b")

    first = shard_ids(ids, replicas=3)

    assert first == shard_ids(ids, replicas=3)
    assert len(first) == 3
    assert Counter(item for shard in first for item in shard) == Counter(ids)


def test_shard_ids_requires_at_least_one_replica() -> None:
    with pytest.raises(ValueError, match="replicas"):
        shard_ids(("example-a",), replicas=0)


def test_generate_one_uses_chat_template_and_greedy_decoding() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()

    result = generate_one(
        model,
        tokenizer,
        system_prompt="System guidance.",
        user_prompt="User request.",
        max_new_tokens=12,
    )

    assert tokenizer.messages == [
        [
            {"role": "system", "content": "System guidance."},
            {"role": "user", "content": "User request."},
        ]
    ]
    assert model.input_ids == [[11, 12, 13]]
    assert model.generate_kwargs == {"max_new_tokens": 12, "do_sample": False}
    assert tokenizer.decoded_tokens == [91, 92]
    assert result.response == "response:91,92"
    assert result.input_tokens == 3
    assert result.generated_tokens == 2
    assert result.used_system_fallback is False


def test_generate_one_folds_system_prompt_when_system_role_is_rejected() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer(rejects_system_role=True)

    result = generate_one(
        model,
        tokenizer,
        system_prompt="System guidance.",
        user_prompt="User request.",
        max_new_tokens=12,
    )

    assert tokenizer.messages == [
        [{"role": "user", "content": "System guidance.\n\nUser request."}]
    ]
    assert result.response == "response:91,92"
    assert result.input_tokens == 3
    assert result.generated_tokens == 2
    assert result.used_system_fallback is True


def test_generate_one_moves_chat_tokens_to_embedding_device() -> None:
    model = DeviceAwareModel()
    tokenizer = DeviceAwareTokenizer()

    generate_one(model, tokenizer, system_prompt="", user_prompt="neutral", max_new_tokens=1)

    assert isinstance(model.input_ids, MovableTokenIds)
    assert model.input_ids.moved_to == "target-device"


def _materialization() -> MaterializationRecord:
    return MaterializationRecord(
        schema_version="reviewer_eval.v1",
        run_id="run:fixture",
        config_hash="a" * 64,
        sample_id="fixture:001",
        source="fixture",
        method="init",
        checkpoint=0,
        system_prompt="system",
        user_prompt="neutral user",
        flat_prompt="neutral user",
        prefix_token_ids=(1,),
        seed_token_ids=(2,),
        prefix_projection_cosine=1.0,
        seed_projection_cosine=1.0,
        semantic_similarity_before=1.0,
        semantic_similarity_after=1.0,
        category_before="category-a",
        category_after="category-a",
        intent_preserved=True,
        projection_attack_score_before=None,
        projection_attack_score_after=None,
        status=RecordStatus.complete,
        failure_kind=None,
        failure_reason=None,
    )


def test_generate_response_record_carries_materialization_identity() -> None:
    record = generate_response_record(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        materialization=_materialization(),
        target_key="target",
        target_revision="local:fixture",
        max_new_tokens=7,
    )

    assert record.status is RecordStatus.complete
    assert record.sample_id == "fixture:001"
    assert record.target_key == "target"
    assert record.checkpoint == 0
    assert record.input_tokens == 3
    assert record.generated_tokens == 2
    assert len(record.prompt_hash) == 64
