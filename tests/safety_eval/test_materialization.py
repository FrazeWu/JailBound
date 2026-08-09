from __future__ import annotations

import pytest
import torch

import benchmark.safety_eval.materialization as materialization_module
from benchmark.safety_eval.materialization import (
    ContinuousCandidate,
    DiscreteCandidate,
    build_materialization_record,
    calibrate_threshold,
    materialize_checkpoint,
    meets_semantic_threshold,
    materialize_continuous_state,
    vocabulary_embedding_sha256,
)
from benchmark.safety_eval.objective import EditableState
from benchmark.safety_eval.prompt_contract import TokenizedEditablePrompt
from benchmark.safety_eval.schema import FailureKind, MaterializationRecord, RecordStatus, TransportType, V2MaterializationRecord
from benchmark.safety_eval.io import canonical_hash


def _state() -> EditableState:
    return EditableState(
        z=torch.tensor([[[1.0, 0.0]]]),
        u=torch.tensor([[[0.0, 1.0]]]),
        z0=torch.zeros(1, 1, 2),
        u0=torch.zeros(1, 1, 2),
    )


def test_threshold_uses_lowest_accepted_positive_at_target_recall() -> None:
    assert calibrate_threshold([0.91, 0.88, 0.85, 0.82, 0.80], target_recall=0.8) == pytest.approx(0.82)


def test_materializer_projects_both_editable_blocks_independently() -> None:
    vocabulary = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    baseline = materialize_continuous_state(_state(), vocabulary)
    changed_z = _state()
    changed_z.z = torch.tensor([[[-1.0, 0.0]]])
    changed_u = _state()
    changed_u.u = torch.tensor([[[1.0, 0.0]]])

    assert baseline.prefix_token_ids == (0,)
    assert baseline.seed_token_ids == (1,)
    assert materialize_continuous_state(changed_z, vocabulary).prefix_token_ids != baseline.prefix_token_ids
    assert materialize_continuous_state(changed_u, vocabulary).seed_token_ids != baseline.seed_token_ids


def test_vocabulary_embedding_sha256_binds_tensor_shape_dtype_and_values() -> None:
    baseline = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    assert vocabulary_embedding_sha256(baseline) == vocabulary_embedding_sha256(baseline.clone())
    assert vocabulary_embedding_sha256(baseline) != vocabulary_embedding_sha256(baseline.to(torch.bfloat16))
    assert vocabulary_embedding_sha256(baseline) != vocabulary_embedding_sha256(baseline[:, :1])


def test_materializer_rejects_forbidden_projection_ids() -> None:
    vocabulary = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="no allowed vocabulary"):
        materialize_continuous_state(_state(), vocabulary, forbidden_token_ids=(0, 1))


def test_discrete_candidate_builds_a_complete_strict_materialization_record() -> None:
    record = build_materialization_record(
        schema_version="reviewer_eval.v1",
        run_id="run:neutral",
        config_hash="a" * 64,
        sample_id="sample:neutral",
        source="neutral-source",
        method="neutral-method",
        checkpoint=25,
        system_prompt="system context",
        user_prompt="user context",
        flat_prompt="flat context",
        semantic_similarity_before=1.0,
        semantic_similarity_after=0.9,
        semantic_threshold=0.8,
        category_before="category-a",
        category_after="category-a",
        candidate=DiscreteCandidate(prefix_token_ids=(4,), seed_token_ids=(5,)),
    )

    assert meets_semantic_threshold(0.9, threshold=0.8)
    assert isinstance(record, MaterializationRecord)
    assert record.status is RecordStatus.complete
    assert record.failure_kind is None and record.failure_reason is None
    assert record.prefix_token_ids == (4,)
    assert record.seed_token_ids == (5,)
    assert record.prefix_projection_cosine == record.seed_projection_cosine == 1.0
    assert record.intent_preserved is True


@pytest.mark.parametrize(
    (
        "candidate",
        "flat_prompt",
        "semantic_similarity_after",
        "special_token_ids",
        "failure_kind",
        "failure_reason",
    ),
    [
        (DiscreteCandidate(prefix_token_ids=(), seed_token_ids=()), "flat context", 0.9, (), FailureKind.materialization, "empty candidate"),
        (DiscreteCandidate(prefix_token_ids=(4,), seed_token_ids=(5,)), "", 0.9, (), FailureKind.materialization, "empty candidate"),
        (DiscreteCandidate(prefix_token_ids=(0,), seed_token_ids=(1,)), "flat context", 0.9, (0, 1), FailureKind.materialization, "special-only candidate"),
        (DiscreteCandidate(prefix_token_ids=(4,), seed_token_ids=(5,)), "flat context", 0.7, (), FailureKind.semantic_filter, "below semantic threshold"),
    ],
)
def test_record_builder_returns_typed_failures_for_rejected_candidates(
    candidate: DiscreteCandidate,
    flat_prompt: str,
    semantic_similarity_after: float,
    special_token_ids: tuple[int, ...],
    failure_kind: FailureKind,
    failure_reason: str,
) -> None:
    record = build_materialization_record(
        schema_version="reviewer_eval.v1",
        run_id="run:neutral",
        config_hash="a" * 64,
        sample_id="sample:neutral",
        source="neutral-source",
        method="neutral-method",
        checkpoint=25,
        system_prompt="system context",
        user_prompt="user context",
        flat_prompt=flat_prompt,
        semantic_similarity_before=1.0,
        semantic_similarity_after=semantic_similarity_after,
        semantic_threshold=0.8,
        category_before="category-a",
        category_after="category-a",
        candidate=candidate,
        special_token_ids=special_token_ids,
    )

    assert record.status is RecordStatus.failed
    assert record.failure_kind is failure_kind
    assert record.failure_reason == failure_reason
    assert record.intent_preserved is False


def test_continuous_candidate_projects_both_blocks_into_a_complete_record() -> None:
    record = build_materialization_record(
        schema_version="reviewer_eval.v1",
        run_id="run:neutral",
        config_hash="a" * 64,
        sample_id="sample:neutral",
        source="neutral-source",
        method="neutral-method",
        checkpoint=25,
        system_prompt="system context",
        user_prompt="user context",
        flat_prompt="flat context",
        semantic_similarity_before=1.0,
        semantic_similarity_after=0.9,
        semantic_threshold=0.8,
        category_before="category-a",
        category_after="category-a",
        candidate=ContinuousCandidate(
            state=_state(),
            vocabulary_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
    )

    assert record.status is RecordStatus.complete
    assert (record.prefix_token_ids, record.seed_token_ids) == ((0,), (1,))
    assert record.prefix_projection_cosine == pytest.approx(1.0)
    assert record.seed_projection_cosine == pytest.approx(1.0)


class _Tokenizer:
    all_special_ids = [0]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return " ".join(f"token-{token_id}" for token_id in token_ids if token_id != 0)


def test_v2_materialization_scatters_u_and_decodes_complete_sequence_once() -> None:
    class RecordingTokenizer:
        all_special_ids: list[int] = []

        def __init__(self) -> None:
            self.decode_calls: list[tuple[int, ...]] = []

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            self.decode_calls.append(tuple(token_ids))
            return "decoded"

    prompt = TokenizedEditablePrompt(
        full_text="abcd",
        base_token_ids=torch.tensor([[10, 11, 12, 13]]),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        editable_positions=(1, 3),
        frozen_positions=(0, 2),
        token_offsets=((0, 1), (1, 2), (2, 3), (3, 4)),
        boundary_expansions=((1, 2),),
        tokenizer_revision="fixture-r1",
    )
    tokenizer = RecordingTokenizer()

    result = materialization_module.materialize_v2_candidate(
        candidate=DiscreteCandidate(prefix_token_ids=(20, 21), seed_token_ids=(30, 31)),
        prompt=prompt,
        tokenizer=tokenizer,
    )

    assert result.reconstructed_base_token_ids == (10, 30, 12, 31)
    assert result.complete_token_ids == (20, 21, 10, 30, 12, 31)
    assert tokenizer.decode_calls == [(20, 21, 10, 30, 12, 31)]
    assert result.frozen_positions_unchanged is True


def test_v2_materialization_record_rejects_embedding_transport_and_tampered_hash() -> None:
    payload = {
        "schema_version": "reviewer_eval.v2", "run_id": "run:fixture", "config_hash": "a" * 64,
        "sample_id": "s:1", "source": "s", "method": "method", "branch": "method", "step": 100,
        "transport": TransportType.text, "state_sha256": "b" * 64, "surrogate_tokenizer_sha256": "c" * 64, "surrogate_embedding_sha256": "d" * 64,
        "editable_positions": (1,), "original_token_ids": (3, 4), "projected_z_token_ids": (7,),
        "projected_u_token_ids": (8,), "reconstructed_base_token_ids": (3, 8), "complete_token_ids": (7, 3, 8),
        "frozen_positions_unchanged": True, "span_boundary_expansions": ((0, 1),),
        "full_prompt_similarity": 0.5, "editable_span_similarity": 0.0, "flat_prompt": "fixture",
        "status": RecordStatus.complete, "failure_kind": None, "failure_reason": None,
    }
    record = V2MaterializationRecord.model_validate({
        **payload, "materialization_sha256": canonical_hash(payload),
    })
    assert record.materialization_sha256 == canonical_hash(payload)

    with pytest.raises(ValueError, match="text transport"):
        V2MaterializationRecord.model_validate({**record.model_dump(mode="json"), "transport": TransportType.embedding})
    with pytest.raises(ValueError, match="materialization sha256"):
        V2MaterializationRecord.model_validate({**record.model_dump(mode="json"), "flat_prompt": "tampered"})


def test_v2_materialization_rejects_special_projection_and_invalid_batches() -> None:
    prompt = TokenizedEditablePrompt(
        full_text="ab", base_token_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long), editable_positions=(1,),
        frozen_positions=(0,), token_offsets=((0, 1), (1, 2)),
        boundary_expansions=((1, 2),), tokenizer_revision="fixture",
    )
    result = materialization_module.materialize_v2_candidate(
            candidate=ContinuousCandidate(
                state=EditableState(torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[1.0, 0.0]]]), torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)),
                vocabulary_embeddings=torch.eye(2),
            ), prompt=prompt, tokenizer=_Tokenizer(), special_token_ids=(0,),
        )
    assert 0 not in result.projected_z_token_ids + result.projected_u_token_ids
    with pytest.raises(ValueError, match="batch size 1"):
        materialization_module.materialize_continuous_state(
            EditableState(torch.ones(2, 1, 2), torch.ones(2, 1, 2), torch.ones(2, 1, 2), torch.ones(2, 1, 2)), torch.eye(2)
        )


def test_materialize_checkpoint_prefers_discrete_token_ids_and_retains_checkpoint_identity() -> None:
    payload = {
        "z": torch.tensor([[[1.0, 0.0]]]),
        "u": torch.tensor([[[0.0, 1.0]]]),
        "z_token_ids": torch.tensor([[2]]),
        "u_token_ids": torch.tensor([[3]]),
    }

    record = materialize_checkpoint(
        state_payload=payload,
        vocabulary_embeddings=torch.eye(4, 2),
        tokenizer=_Tokenizer(),
        schema_version="reviewer_eval.v1",
        run_id="run:fixture",
        config_hash="a" * 64,
        sample_id="fixture:001",
        source="fixture",
        method="gcg",
        checkpoint=25,
        original_prompt="neutral fixture",
        category="category-a",
        semantic_similarity=0.95,
        semantic_threshold=0.9,
    )

    assert record.status is RecordStatus.complete
    assert record.checkpoint == 25
    assert record.prefix_token_ids == (2,)
    assert record.seed_token_ids == (3,)
    assert record.user_prompt == record.flat_prompt
    assert "token-2" in record.flat_prompt and "token-3" in record.flat_prompt


def test_materialize_checkpoint_moves_continuous_cpu_state_to_vocabulary_device(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_devices: list[torch.device] = []

    def fake_materialize(state: EditableState, vocabulary: torch.Tensor, **_: object) -> object:
        assert state.z.device == vocabulary.device
        assert state.u.device == vocabulary.device
        observed_devices.append(vocabulary.device)
        return materialization_module.ContinuousMaterialization((2,), (3,), 1.0, 1.0)

    monkeypatch.setattr(materialization_module, "materialize_continuous_state", fake_materialize)
    payload = {
        "z": torch.tensor([[[1.0, 0.0]]]),
        "u": torch.tensor([[[0.0, 1.0]]]),
        "z_token_ids": torch.empty((0, 0), dtype=torch.long),
        "u_token_ids": torch.empty((0, 0), dtype=torch.long),
    }

    record = materialize_checkpoint(
        state_payload=payload,
        vocabulary_embeddings=torch.empty((4, 2), device="meta"),
        tokenizer=_Tokenizer(),
        schema_version="reviewer_eval.v1",
        run_id="run:fixture",
        config_hash="a" * 64,
        sample_id="fixture:001",
        source="fixture",
        method="continuous",
        checkpoint=100,
        original_prompt="neutral fixture",
        category="category-a",
        semantic_similarity=0.95,
        semantic_threshold=0.9,
    )

    assert record.status is RecordStatus.complete
    assert observed_devices == [torch.device("meta"), torch.device("meta")]
