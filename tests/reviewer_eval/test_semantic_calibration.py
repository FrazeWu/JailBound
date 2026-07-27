from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from benchmark.reviewer_eval.semantic import (
    CalibrationCandidate,
    CalibrationPair,
    QwenHiddenMeanEncoder,
    build_calibration_pairs,
    freeze_semantic_calibration,
    select_calibration_holdouts,
)


def test_freeze_semantic_calibration_uses_positive_recall_rule_and_records_negative_acceptance() -> None:
    artifact = freeze_semantic_calibration(
        (
            CalibrationPair("a:1", "a", True, 0.95),
            CalibrationPair("a:2", "a", True, 0.80),
            CalibrationPair("b:1", "b", True, 0.70),
            CalibrationPair("a:n", "a", False, 0.90),
            CalibrationPair("b:n", "b", False, 0.20),
        ),
        target_recall=2 / 3,
        encoder_revision="local:fixture",
    )

    assert artifact["threshold"] == pytest.approx(0.80)
    assert artifact["positive_recall"] == pytest.approx(2 / 3)
    assert artifact["negative_acceptance"] == pytest.approx(1 / 2)
    assert artifact["encoder_revision"] == "local:fixture"
    assert artifact["pair_count"] == 5


def test_freeze_semantic_calibration_rejects_duplicate_pair_ids() -> None:
    pairs = (
        CalibrationPair("a:1", "a", True, 0.9),
        CalibrationPair("a:1", "a", False, 0.2),
    )

    with pytest.raises(ValueError, match="duplicate"):
        freeze_semantic_calibration(pairs, target_recall=0.95, encoder_revision="local:fixture")


def test_hidden_mean_encoder_reuses_an_injected_loaded_model() -> None:
    class Batch(dict[str, torch.Tensor]):
        def to(self, device: object) -> "Batch":
            assert device == "cpu"
            return self

    class Tokenizer:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, texts: list[str], **_: object) -> Batch:
            self.calls += 1
            return Batch(
                input_ids=torch.tensor([[1], [2]]),
                attention_mask=torch.tensor([[1], [1]]),
            )

    class Model:
        device = "cpu"

        def __init__(self) -> None:
            self.calls = 0
            self.evaluated = False

        def eval(self) -> "Model":
            self.evaluated = True
            return self

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            self.calls += 1
            assert kwargs["output_hidden_states"] is True
            return SimpleNamespace(hidden_states=(torch.tensor([[[3.0, 4.0]], [[0.0, 5.0]]]),))

    tokenizer = Tokenizer()
    model = Model()
    encoder = QwenHiddenMeanEncoder(
        "/not-used-when-injected",
        tokenizer=tokenizer,
        model=model,
        revision="local:fixture",
    )

    vectors = encoder.encode(["one", "two"])

    assert tokenizer.calls == model.calls == 1
    assert model.evaluated is True
    assert encoder.resolved_revision == "local:fixture"
    assert vectors.shape == (2, 2)
    assert vectors[0].tolist() == pytest.approx([0.6, 0.8])


def test_calibration_holdouts_are_disjoint_and_have_same_category_negative_partners() -> None:
    candidates = (
        CalibrationCandidate("source:001", "source", "category_a", "intent 1"),
        CalibrationCandidate("source:002", "source", "category_a", "intent 2"),
        CalibrationCandidate("source:003", "source", "category_b", "intent 3"),
        CalibrationCandidate("source:004", "source", "category_b", "intent 4"),
        CalibrationCandidate("source:005", "source", "category_c", "intent 5"),
        CalibrationCandidate("source:006", "source", "category_c", "intent 6"),
        CalibrationCandidate("source:007", "source", "category_d", "intent 7"),
        CalibrationCandidate("source:008", "source", "category_d", "intent 8"),
    )

    selected = select_calibration_holdouts(
        candidates,
        controlled_ids=frozenset({"source:001", "source:004"}),
        per_source=4,
        seed=20260725,
    )

    assert len(selected) == 4
    assert {item.candidate.example_id for item in selected}.isdisjoint({"source:001", "source:004"})
    assert all(item.candidate.risk_category == item.negative.risk_category for item in selected)
    assert all(item.candidate.example_id != item.negative.example_id for item in selected)


def test_calibration_pairs_retry_until_semantic_constraints_are_met() -> None:
    candidate = CalibrationCandidate("source:001", "source", "category_a", "intent one")
    negative = CalibrationCandidate("source:002", "source", "category_a", "intent two")
    attempts = iter(("wrong category", "accepted rewrite"))

    pairs = build_calibration_pairs(
        (select_calibration_holdouts((candidate, negative), controlled_ids=frozenset(), per_source=1, seed=7)[0],),
        paraphrase=lambda _: next(attempts),
        category_for_text=lambda text: "category_b" if text == "wrong category" else "category_a",
        entities_preserved=lambda _, __: True,
        similarity=lambda left, right: 0.9 if right == "accepted rewrite" else 0.1,
        max_attempts=2,
    )

    assert [(pair.positive, pair.similarity) for pair in pairs] == [(True, 0.9), (False, 0.1)]
    assert len({pair.pair_id for pair in pairs}) == 2
