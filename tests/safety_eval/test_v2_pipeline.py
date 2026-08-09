from __future__ import annotations

from pathlib import Path

import pytest
import torch

from benchmark.safety_eval.io import canonical_hash, sha256_file
from benchmark.safety_eval.schema import OptimizationRecord, V2BenchmarkExample
from benchmark.safety_eval.v2_pipeline import (
    materialize_v2_optimization_state,
    materialize_v2_terminal_records,
)
from benchmark.safety_eval.materialization import vocabulary_embedding_sha256
from benchmark.safety_eval.runner import OptimizationJob, stable_state_id


class Tokenizer:
    all_special_ids: list[int] = []

    def __call__(self, text: str, *, return_offsets_mapping: bool) -> dict[str, object]:
        assert text == "abc"
        assert return_offsets_mapping is True
        return {
            "input_ids": [[0, 1, 2]],
            "attention_mask": [[1, 1, 1]],
            "offset_mapping": [[(0, 1), (1, 2), (2, 3)]],
        }

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "".join("abc"[value] for value in ids)


class SizedTokenizer(Tokenizer):
    def __len__(self) -> int:
        return 3

    def get_vocab(self) -> dict[str, int]:
        return {"a": 0, "b": 1, "c": 2}


class InvalidMappingTokenizer(SizedTokenizer):
    def get_vocab(self) -> dict[str, int]:
        return {"a": 0, "b": 2, "c": 3}


def test_v2_adapter_rejects_checkpoint_from_a_different_input_embedding_matrix(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.pt"
    embedding_used_for_optimization = torch.eye(3, 2)
    torch.save(
        {
            "z": torch.tensor([[[1.0, 0.0]]]),
            "u": torch.tensor([[[0.0, 1.0]]]),
            "base_token_ids": torch.tensor([[0, 1, 2]]),
            "editable_positions": torch.tensor([1]),
            "tokenizer_revision": "c" * 64,
            "editable_span_hashes": (
                canonical_hash({"start": 1, "end": 2, "quote": "b", "role": "harmful_payload", "confidence": 1.0, "rationale": "fixture"}),
            ),
            "input_embedding_sha256": vocabulary_embedding_sha256(
                embedding_used_for_optimization
            ),
        },
        state,
    )
    example = V2BenchmarkExample.model_validate({
        "schema_version": "reviewer_eval.v2", "example_id": "s:1", "source": "s", "source_file": "fixture", "source_row": 1, "source_sha256": "a" * 64,
        "intent": "b", "attack_text": "abc", "target_text": None, "source_risk_label": None, "source_attack_label": "direct_request", "risk_category": "risk", "threat_domain": "domain", "attack_type": "direct_request", "language": "en", "selection_stratum": "risk|direct_request", "selection_seed": 1, "prompt_sha256": "b" * 64,
        "preprocessing": [], "intent_sha256": "c" * 64,
        "editable_spans": [{"start": 1, "end": 2, "quote": "b", "role": "harmful_payload", "confidence": 1.0, "rationale": "fixture"}], "annotator_model": "model", "annotator_revision": "revision", "annotation_template_sha256": "d" * 64, "annotation_response_sha256": "e" * 64, "annotation_confidence": 1.0,
    })
    optimization = OptimizationRecord.model_validate({
        "schema_version": "reviewer_eval.v2", "run_id": "run:fixture", "config_hash": "f" * 64, "git_revision": "fixture", "cell_id": "cell:fixture", "sample_id": "s:1", "source": "s", "method": "random_mutation", "checkpoint": 100, "random_seed": 1, "status": "complete", "failure_kind": None, "failure_reason": None, "state_path": str(state), "state_sha256": sha256_file(state), "representation": "tensor_embeddings:random_mutation", "attack_loss": None, "fol": None, "internal_margin": None, "materialized_prompt": None, "counters": {},
    })

    with pytest.raises(ValueError, match="input embedding sha256"):
        materialize_v2_optimization_state(
            optimization,
            example=example,
            vocabulary_embeddings=torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
            tokenizer=Tokenizer(),
            surrogate_tokenizer_sha256="c" * 64,
        )


def test_v2_adapter_reconstructs_only_the_annotated_token(tmp_path: Path) -> None:
    state = tmp_path / "state.pt"
    torch.save({"z": torch.tensor([[[1.0, 0.0]]]), "u": torch.tensor([[[0.0, 1.0]]]), "base_token_ids": torch.tensor([[0, 1, 2]]), "editable_positions": torch.tensor([1]), "tokenizer_revision": "c" * 64, "editable_span_hashes": (canonical_hash({"start": 1, "end": 2, "quote": "b", "role": "harmful_payload", "confidence": 1.0, "rationale": "fixture"}),), "input_embedding_sha256": vocabulary_embedding_sha256(torch.eye(3, 2))}, state)
    example = V2BenchmarkExample.model_validate({
        "schema_version": "reviewer_eval.v2", "example_id": "s:1", "source": "s",
        "source_file": "fixture", "source_row": 1, "source_sha256": "a" * 64,
        "intent": "b", "attack_text": "abc", "target_text": None,
        "source_risk_label": None, "source_attack_label": "direct_request",
        "risk_category": "risk", "threat_domain": "domain", "attack_type": "direct_request",
        "language": "en", "selection_stratum": "risk|direct_request", "selection_seed": 1,
        "prompt_sha256": "b" * 64, "preprocessing": [], "intent_sha256": "c" * 64,
        "editable_spans": [{"start": 1, "end": 2, "quote": "b", "role": "harmful_payload", "confidence": 1.0, "rationale": "fixture"}],
        "annotator_model": "model", "annotator_revision": "revision",
        "annotation_template_sha256": "d" * 64, "annotation_response_sha256": "e" * 64,
        "annotation_confidence": 1.0,
    })
    optimization = OptimizationRecord.model_validate({
        "schema_version": "reviewer_eval.v2", "run_id": "run:fixture", "config_hash": "f" * 64,
        "git_revision": "fixture", "cell_id": "cell:fixture", "sample_id": "s:1", "source": "s",
        "method": "random_mutation", "checkpoint": 100, "random_seed": 1,
        "status": "complete", "failure_kind": None, "failure_reason": None, "state_path": str(state),
        "state_sha256": sha256_file(state),
        "representation": "tensor_embeddings:random_mutation", "attack_loss": None, "fol": None,
        "internal_margin": None, "materialized_prompt": None, "counters": {},
    })

    result = materialize_v2_optimization_state(
        optimization, example=example, vocabulary_embeddings=torch.eye(3, 2), tokenizer=Tokenizer(), surrogate_tokenizer_sha256="c" * 64
    )

    assert result.flat_prompt == "aabc"
    assert result.projected_u_token_ids == (1,)
    assert result.reconstructed_base_token_ids == (0, 1, 2)
    assert result.frozen_positions_unchanged is True


def test_v2_adapter_rejects_a_state_file_whose_bytes_no_longer_match_record(tmp_path: Path) -> None:
    state = tmp_path / "state.pt"
    torch.save({"z": torch.tensor([[[1.0, 0.0]]])}, state)
    optimization = OptimizationRecord.model_validate({
        "schema_version": "reviewer_eval.v2", "run_id": "run:fixture", "config_hash": "f" * 64,
        "git_revision": "fixture", "cell_id": "cell:fixture", "sample_id": "s:1", "source": "s",
        "method": "random_mutation", "checkpoint": 100, "random_seed": 1,
        "status": "complete", "failure_kind": None, "failure_reason": None, "state_path": str(state),
        "state_sha256": "0" * 64, "representation": "tensor_embeddings:random_mutation",
        "attack_loss": None, "fol": None, "internal_margin": None, "materialized_prompt": None, "counters": {},
    })
    example = V2BenchmarkExample.model_validate({
        "schema_version": "reviewer_eval.v2", "example_id": "s:1", "source": "s", "source_file": "fixture",
        "source_row": 1, "source_sha256": "a" * 64, "intent": "b", "attack_text": "abc", "target_text": None,
        "source_risk_label": None, "source_attack_label": "direct_request", "risk_category": "risk",
        "threat_domain": "domain", "attack_type": "direct_request", "language": "en",
        "selection_stratum": "risk|direct_request", "selection_seed": 1, "prompt_sha256": "b" * 64,
        "preprocessing": [], "intent_sha256": "c" * 64,
        "editable_spans": [{"start": 1, "end": 2, "quote": "b", "role": "harmful_payload", "confidence": 1.0, "rationale": "fixture"}],
        "annotator_model": "model", "annotator_revision": "revision", "annotation_template_sha256": "d" * 64,
        "annotation_response_sha256": "e" * 64, "annotation_confidence": 1.0,
    })

    with pytest.raises(ValueError, match="state sha256"):
        materialize_v2_optimization_state(
            optimization, example=example, vocabulary_embeddings=torch.eye(3, 2), tokenizer=Tokenizer(), surrogate_tokenizer_sha256="c" * 64
        )


def test_v2_terminal_materialization_uses_v2_manifest_and_resumes(tmp_path: Path) -> None:
    state = (
        tmp_path / "optimization" / "s" / "random_mutation" / "states"
        / f"{stable_state_id(OptimizationJob('s', 'random_mutation', 'cell:fixture', 's:1', 1), 100)}.pt"
    )
    state.parent.mkdir(parents=True)
    torch.save({"z": torch.tensor([[[1.0, 0.0]]]), "u": torch.tensor([[[0.0, 1.0]]]), "base_token_ids": torch.tensor([[0, 1, 2]]), "editable_positions": torch.tensor([1]), "tokenizer_revision": "c" * 64, "editable_span_hashes": (canonical_hash({"start": 1, "end": 2, "quote": "b", "role": "harmful_payload", "confidence": 1.0, "rationale": "fixture"}),), "input_embedding_sha256": vocabulary_embedding_sha256(torch.eye(3, 2))}, state)
    example = {
        "schema_version": "reviewer_eval.v2", "example_id": "s:1", "source": "s",
        "source_file": "fixture", "source_row": 1, "source_sha256": "a" * 64,
        "intent": "b", "attack_text": "abc", "target_text": None, "source_risk_label": None,
        "source_attack_label": "direct_request", "risk_category": "risk", "threat_domain": "domain",
        "attack_type": "direct_request", "language": "en", "selection_stratum": "risk|direct_request",
        "selection_seed": 1, "prompt_sha256": "b" * 64, "preprocessing": [], "intent_sha256": "c" * 64,
        "editable_spans": [{"start": 1, "end": 2, "quote": "b", "role": "harmful_payload", "confidence": 1.0, "rationale": "fixture"}],
        "annotator_model": "model", "annotator_revision": "revision", "annotation_template_sha256": "d" * 64,
        "annotation_response_sha256": "e" * 64, "annotation_confidence": 1.0,
    }
    manifest = tmp_path / "manifests" / "v2"
    manifest.mkdir(parents=True)
    (manifest / "controlled_s.jsonl").write_text(__import__("json").dumps(example) + "\n", encoding="utf-8")
    optimization = {
        "schema_version": "reviewer_eval.v2", "run_id": "run:fixture", "config_hash": "f" * 64,
        "git_revision": "fixture", "cell_id": "cell:fixture", "sample_id": "s:1", "source": "s",
        "method": "random_mutation", "checkpoint": 100, "random_seed": 1, "status": "complete",
        "failure_kind": None, "failure_reason": None, "state_path": str(state),
        "state_sha256": sha256_file(state),
        "representation": "tensor_embeddings:random_mutation", "attack_loss": None, "fol": None,
        "internal_margin": None, "materialized_prompt": None, "counters": {},
    }
    records = tmp_path / "optimization" / "s" / "random_mutation"
    records.mkdir(parents=True, exist_ok=True)
    (records / "records.jsonl").write_text(__import__("json").dumps(optimization) + "\n", encoding="utf-8")

    first = materialize_v2_terminal_records(tmp_path, source="s", method="random_mutation", vocabulary_embeddings=torch.eye(3, 2), tokenizer=Tokenizer(), surrogate_tokenizer_sha256="c" * 64)
    second = materialize_v2_terminal_records(tmp_path, source="s", method="random_mutation", vocabulary_embeddings=torch.eye(3, 2), tokenizer=Tokenizer(), surrogate_tokenizer_sha256="c" * 64)

    assert len(first) == len(second) == 1
    assert len((records / "materialization.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_v2_terminal_materialization_rejects_conflicting_resume_payload(tmp_path: Path) -> None:
    """A matching resume key cannot silently substitute a different projection."""
    # Reuse the compact fixture above, then replace its ledger row with a
    # self-hashed but semantically different materialization under the same key.
    test_v2_terminal_materialization_uses_v2_manifest_and_resumes(tmp_path)
    ledger = tmp_path / "optimization" / "s" / "random_mutation" / "materialization.jsonl"
    payload = __import__("json").loads(ledger.read_text(encoding="utf-8"))
    payload["flat_prompt"] = "different"
    payload.pop("materialization_sha256")
    payload["materialization_sha256"] = canonical_hash(payload)
    ledger.write_text(__import__("json").dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting payload"):
        materialize_v2_terminal_records(
            tmp_path,
            source="s",
            method="random_mutation",
            vocabulary_embeddings=torch.eye(3, 2),
            tokenizer=Tokenizer(),
            surrogate_tokenizer_sha256="c" * 64,
        )


def test_v2_materialization_rejects_hard_token_ids_that_disagree_with_projection(tmp_path: Path) -> None:
    from benchmark.safety_eval.v2_pipeline import _validate_discrete_projection

    with pytest.raises(ValueError, match="re-projection"):
        _validate_discrete_projection(
            {"z_token_ids": torch.tensor([[2]]), "u_token_ids": torch.tensor([[2]])},
            (0,), (1,),
        )


def test_v2_materialization_rejects_embedding_rows_that_do_not_match_tokenizer(tmp_path: Path) -> None:
    state = tmp_path / "state.pt"
    torch.save(
        {
            "z": torch.tensor([[[1.0, 0.0]]]),
            "u": torch.tensor([[[0.0, 1.0]]]),
            "base_token_ids": torch.tensor([[0, 1, 2]]),
            "editable_positions": torch.tensor([1]),
            "tokenizer_revision": "c" * 64,
            "editable_span_hashes": (
                canonical_hash({"start": 1, "end": 2, "quote": "b", "role": "harmful_payload", "confidence": 1.0, "rationale": "fixture"}),
            ),
        },
        state,
    )
    example = V2BenchmarkExample.model_validate({
        "schema_version": "reviewer_eval.v2", "example_id": "s:1", "source": "s", "source_file": "fixture", "source_row": 1, "source_sha256": "a" * 64,
        "intent": "b", "attack_text": "abc", "target_text": None, "source_risk_label": None, "source_attack_label": "direct_request", "risk_category": "risk", "threat_domain": "domain", "attack_type": "direct_request", "language": "en", "selection_stratum": "risk|direct_request", "selection_seed": 1, "prompt_sha256": "b" * 64, "preprocessing": [], "intent_sha256": "c" * 64,
        "editable_spans": [{"start": 1, "end": 2, "quote": "b", "role": "harmful_payload", "confidence": 1.0, "rationale": "fixture"}], "annotator_model": "model", "annotator_revision": "revision", "annotation_template_sha256": "d" * 64, "annotation_response_sha256": "e" * 64, "annotation_confidence": 1.0,
    })
    optimization = OptimizationRecord.model_validate({
        "schema_version": "reviewer_eval.v2", "run_id": "run:fixture", "config_hash": "f" * 64, "git_revision": "fixture", "cell_id": "cell:fixture", "sample_id": "s:1", "source": "s", "method": "random_mutation", "checkpoint": 100, "random_seed": 1, "status": "complete", "failure_kind": None, "failure_reason": None, "state_path": str(state), "state_sha256": sha256_file(state), "representation": "tensor_embeddings:random_mutation", "attack_loss": None, "fol": None, "internal_margin": None, "materialized_prompt": None, "counters": {},
    })

    with pytest.raises(ValueError, match="embedding rows"):
        materialize_v2_optimization_state(
            optimization,
            example=example,
            vocabulary_embeddings=torch.eye(2),
            tokenizer=SizedTokenizer(),
            surrogate_tokenizer_sha256="c" * 64,
        )


def test_v2_materialization_rejects_noncontiguous_tokenizer_id_mapping() -> None:
    from benchmark.safety_eval.v2_pipeline import _validate_vocabulary_embedding_contract

    with pytest.raises(ValueError, match="contiguous"):
        _validate_vocabulary_embedding_contract(torch.eye(3), InvalidMappingTokenizer())


def test_v2_materialization_allows_unmapped_reserved_embedding_rows() -> None:
    from benchmark.safety_eval.v2_pipeline import _validate_vocabulary_embedding_contract

    _validate_vocabulary_embedding_contract(torch.eye(4), SizedTokenizer())


def test_v2_adapter_retains_saved_bfloat16_hard_ids_when_float32_projection_ties(tmp_path: Path) -> None:
    state = tmp_path / "state.pt"
    span_hash = canonical_hash({"start": 1, "end": 2, "quote": "b", "role": "harmful_payload", "confidence": 1.0, "rationale": "fixture"})
    torch.save({
        "z": torch.tensor([[[1.0, 0.0]]], dtype=torch.bfloat16),
        "u": torch.tensor([[[0.0, 1.0]]], dtype=torch.bfloat16),
        "z_token_ids": torch.tensor([[1]]), "u_token_ids": torch.tensor([[2]]),
        "base_token_ids": torch.tensor([[0, 1, 2]]), "editable_positions": torch.tensor([1]),
        "tokenizer_revision": "c" * 64, "editable_span_hashes": (span_hash,),
        "input_embedding_sha256": vocabulary_embedding_sha256(torch.tensor([[1.0, 0.0], [1.0001, 0.0], [0.0, 1.0]])),
    }, state)
    example = V2BenchmarkExample.model_validate({
        "schema_version": "reviewer_eval.v2", "example_id": "s:1", "source": "s", "source_file": "fixture", "source_row": 1,
        "source_sha256": "a" * 64, "intent": "b", "attack_text": "abc", "target_text": None, "source_risk_label": None,
        "source_attack_label": "direct_request", "risk_category": "risk", "threat_domain": "domain", "attack_type": "direct_request",
        "language": "en", "selection_stratum": "risk|direct_request", "selection_seed": 1, "prompt_sha256": "b" * 64,
        "preprocessing": [], "intent_sha256": "c" * 64,
        "editable_spans": [{"start": 1, "end": 2, "quote": "b", "role": "harmful_payload", "confidence": 1.0, "rationale": "fixture"}],
        "annotator_model": "model", "annotator_revision": "revision", "annotation_template_sha256": "d" * 64,
        "annotation_response_sha256": "e" * 64, "annotation_confidence": 1.0,
    })
    optimization = OptimizationRecord.model_validate({
        "schema_version": "reviewer_eval.v2", "run_id": "run:fixture", "config_hash": "f" * 64, "git_revision": "fixture", "cell_id": "cell:fixture",
        "sample_id": "s:1", "source": "s", "method": "pez", "checkpoint": 100, "random_seed": 1, "status": "complete",
        "failure_kind": None, "failure_reason": None, "state_path": str(state), "state_sha256": sha256_file(state),
        "representation": "tensor_embeddings:pez", "attack_loss": None, "fol": None, "internal_margin": None, "materialized_prompt": None, "counters": {},
    })

    result = materialize_v2_optimization_state(
        optimization, example=example,
        # bfloat16 rounded the saved ID-1 vector to [1, 0].  In float32 both
        # ID 0 and ID 1 are collinear, so a second argmax selects ID 0.
        vocabulary_embeddings=torch.tensor([[1.0, 0.0], [1.0001, 0.0], [0.0, 1.0]]),
        tokenizer=Tokenizer(), surrogate_tokenizer_sha256="c" * 64,
    )

    assert result.projected_z_token_ids == (1,)
    assert result.projected_u_token_ids == (2,)


def test_v2_adapter_rejects_hard_ids_whose_state_is_not_their_embedding(tmp_path: Path) -> None:
    from benchmark.safety_eval.v2_pipeline import _validate_saved_hard_token_state

    with pytest.raises(ValueError, match="hard-token embedding"):
        _validate_saved_hard_token_state(
            {
                "z": torch.tensor([[[0.0, 1.0]]]), "u": torch.tensor([[[0.0, 1.0]]]),
                "z_token_ids": torch.tensor([[1]]), "u_token_ids": torch.tensor([[2]]),
            },
            vocabulary_embeddings=torch.eye(3, 2), forbidden_token_ids=(),
        )


def test_v2_adapter_rejects_float32_hard_state_that_only_fits_bfloat16_tolerance() -> None:
    from benchmark.safety_eval.v2_pipeline import _validate_saved_hard_token_state

    with pytest.raises(ValueError, match="hard-token embedding"):
        _validate_saved_hard_token_state(
            {
                "z": torch.tensor([[[1.01, 0.0]]]), "u": torch.tensor([[[0.0, 1.0]]]),
                "z_token_ids": torch.tensor([[0]]), "u_token_ids": torch.tensor([[1]]),
            },
            vocabulary_embeddings=torch.eye(2, dtype=torch.bfloat16), forbidden_token_ids=(),
        )


def test_v2_adapter_rejects_complex_hard_token_ids() -> None:
    from benchmark.safety_eval.v2_pipeline import _validate_saved_hard_token_state

    with pytest.raises(ValueError, match="integer"):
        _validate_saved_hard_token_state(
            {
                "z": torch.tensor([[[1.0, 0.0]]]), "u": torch.tensor([[[0.0, 1.0]]]),
                "z_token_ids": torch.tensor([[0 + 0j]]), "u_token_ids": torch.tensor([[1 + 0j]]),
            },
            vocabulary_embeddings=torch.eye(2), forbidden_token_ids=(),
        )
