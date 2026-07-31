"""V2 post-optimization adapters that preserve annotated editable spans."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .materialization import (
    ContinuousCandidate,
    build_v2_materialization_record,
    materialize_v2_candidate,
)
from .objective import EditableState
from .prompt_contract import tokenize_editable_prompt
from .schema import OptimizationRecord, V2BenchmarkExample, V2MaterializationRecord


def materialize_v2_optimization_state(
    optimization: OptimizationRecord,
    *,
    example: V2BenchmarkExample,
    vocabulary_embeddings: torch.Tensor,
    tokenizer: Any,
) -> V2MaterializationRecord:
    """Project one terminal single-branch state without modifying frozen tokens."""
    if optimization.schema_version != "reviewer_eval.v2":
        raise ValueError("v2 materialization requires a v2 optimization record")
    if (optimization.sample_id, optimization.source) != (
        example.example_id,
        example.source,
    ):
        raise ValueError("optimization record does not match its immutable example")
    if not optimization.state_path:
        raise ValueError("v2 optimization state path is required")
    payload = torch.load(Path(optimization.state_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("v2 optimization state must be a mapping")
    z, u = payload.get("z"), payload.get("u")
    if not isinstance(z, torch.Tensor) or not isinstance(u, torch.Tensor):
        raise ValueError("v2 optimization state requires z and u tensors")
    prompt = tokenize_editable_prompt(
        example.attack_text, example.editable_spans, tokenizer, optimization.sample_id
    )
    z = z.to(vocabulary_embeddings.device)
    u = u.to(vocabulary_embeddings.device)
    result = materialize_v2_candidate(
        candidate=ContinuousCandidate(
            state=EditableState(z=z, u=u, z0=z.detach().clone(), u0=u.detach().clone()),
            vocabulary_embeddings=vocabulary_embeddings,
        ),
        prompt=prompt,
        tokenizer=tokenizer,
    )
    branch = optimization.representation.rsplit(":", 1)[-1]
    return build_v2_materialization_record(
        result=result,
        prompt=prompt,
        run_id=optimization.run_id,
        config_hash=optimization.config_hash,
        sample_id=optimization.sample_id,
        source=optimization.source,
        method=optimization.method,
        branch=branch,
        step=optimization.checkpoint,
    )
