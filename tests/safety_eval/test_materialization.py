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
)
from benchmark.safety_eval.objective import EditableState
from benchmark.safety_eval.schema import FailureKind, MaterializationRecord, RecordStatus


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


def test_materializer_rejects_forbidden_projection_ids() -> None:
    vocabulary = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="no allowed vocabulary"):
        materialize_continuous_state(_state(), vocabulary, forbidden_token_ids=(0, 1))


def test_materializer_projects_both_blocks_only_into_explicit_allowed_ids() -> None:
    vocabulary = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])

    materialization = materialize_continuous_state(
        _state(), vocabulary, allowed_token_ids=(1, 3)
    )

    assert materialization.prefix_token_ids == (1,)
    assert materialization.seed_token_ids == (3,)


@pytest.mark.parametrize("allowed_token_ids", [(), (4,), (1, 1)])
def test_materializer_rejects_invalid_explicit_allowed_token_ids(
    allowed_token_ids: tuple[int, ...],
) -> None:
    vocabulary = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="allowed token"):
        materialize_continuous_state(
            _state(), vocabulary, allowed_token_ids=allowed_token_ids
        )


def test_materializer_projects_distinct_position_masks_for_both_blocks() -> None:
    vocabulary = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    state = EditableState(
        z=torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]]),
        u=torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]]),
        z0=torch.zeros(1, 2, 2),
        u0=torch.zeros(1, 2, 2),
    )

    materialization = materialize_continuous_state(
        state,
        vocabulary,
        prefix_allowed_token_ids_by_position=((1,), (2, 3)),
        seed_allowed_token_ids_by_position=((0, 1), (3,)),
    )

    assert materialization.prefix_token_ids == (1, 2)
    assert materialization.seed_token_ids == (0, 3)


@pytest.mark.parametrize(
    "position_masks",
    [
        {"prefix_allowed_token_ids_by_position": ((0,),)},
        {"seed_allowed_token_ids_by_position": ((1,),)},
    ],
)
def test_materializer_rejects_global_and_position_mask_conflicts(
    position_masks: dict[str, tuple[tuple[int, ...], ...]],
) -> None:
    vocabulary = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="global and position"):
        materialize_continuous_state(
            _state(), vocabulary, allowed_token_ids=(0, 1), **position_masks
        )


@pytest.mark.parametrize(
    "position_masks",
    [
        {"prefix_allowed_token_ids_by_position": ((0,),)},
        {"seed_allowed_token_ids_by_position": ((1,),)},
    ],
)
def test_materializer_requires_position_masks_for_both_blocks(
    position_masks: dict[str, tuple[tuple[int, ...], ...]],
) -> None:
    vocabulary = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="both prefix and seed"):
        materialize_continuous_state(_state(), vocabulary, **position_masks)


@pytest.mark.parametrize(
    "position_masks",
    [
        {
            "prefix_allowed_token_ids_by_position": ((0,), (1,)),
            "seed_allowed_token_ids_by_position": ((1,),),
        },
        {
            "prefix_allowed_token_ids_by_position": ((0,),),
            "seed_allowed_token_ids_by_position": ((1,), (0,)),
        },
    ],
)
def test_materializer_rejects_wrong_position_mask_count(
    position_masks: dict[str, tuple[tuple[int, ...], ...]],
) -> None:
    vocabulary = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="position mask count"):
        materialize_continuous_state(_state(), vocabulary, **position_masks)


@pytest.mark.parametrize(
    ("invalid_mask", "error"),
    [
        ((), "must not be empty"),
        ((0, 0), "must be unique"),
        ((2,), "must be in range"),
        ((1,), "must not be forbidden"),
    ],
)
def test_materializer_rejects_invalid_position_masks(
    invalid_mask: tuple[int, ...], error: str
) -> None:
    vocabulary = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match=error):
        materialize_continuous_state(
            _state(),
            vocabulary,
            forbidden_token_ids=(1,),
            prefix_allowed_token_ids_by_position=(invalid_mask,),
            seed_allowed_token_ids_by_position=((0,),),
        )


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
