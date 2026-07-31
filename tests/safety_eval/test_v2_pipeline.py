from __future__ import annotations

from pathlib import Path

import torch

from benchmark.safety_eval.schema import OptimizationRecord, V2BenchmarkExample
from benchmark.safety_eval.v2_pipeline import materialize_v2_optimization_state


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


def test_v2_adapter_reconstructs_only_the_annotated_token(tmp_path: Path) -> None:
    state = tmp_path / "state.pt"
    torch.save({"z": torch.tensor([[[1.0, 0.0]]]), "u": torch.tensor([[[0.0, 1.0]]])}, state)
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
        "representation": "tensor_embeddings:random_mutation", "attack_loss": None, "fol": None,
        "internal_margin": None, "materialized_prompt": None, "counters": {},
    })

    result = materialize_v2_optimization_state(
        optimization, example=example, vocabulary_embeddings=torch.eye(3, 2), tokenizer=Tokenizer()
    )

    assert result.flat_prompt == "aabc"
    assert result.projected_u_token_ids == (1,)
    assert result.reconstructed_base_token_ids == (0, 1, 2)
    assert result.frozen_positions_unchanged is True
