"""Compute content-free FOL control covariates for the frozen validation prompts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.execution import _embedding_device, load_local_qwen
from benchmark.safety_eval.fol_records import resolved_terminal_payloads
from benchmark.safety_eval.fol_boundary import random_joint_direction
from benchmark.safety_eval.fol_runtime import causal_perplexity
from benchmark.safety_eval.io import JsonlLedger, canonical_hash, read_jsonl
from benchmark.safety_eval.objective import EditableState
from benchmark.safety_eval.runtime import validate_model_assets
from benchmark.safety_eval.schema import BenchmarkExample, OptimizationRecord, RecordStatus
from benchmark.safety_eval.transformer_objective import TransformerAttackObjective


ROOT = Path(__file__).resolve().parents[1]


def _selected_sources(configured: tuple[str, ...], requested: list[str] | None) -> tuple[str, ...]:
    sources = tuple(requested or configured)
    if set(sources) - set(configured):
        raise ValueError("FOL controls requested an unconfigured source")
    return sources


def _append_control_rows(path: Path, rows: list[dict[str, object]]) -> int:
    ledger = JsonlLedger(path, key_fields=("source", "sample_id"))
    return sum(ledger.append_once(row) for row in rows)


def _validation_ids(root: Path, source: str) -> set[str]:
    payload = json.loads((root / "manifests" / "validation_selection.json").read_text(encoding="utf-8"))
    source_payload = payload.get("sources", {}).get(source) if isinstance(payload.get("sources"), dict) else None
    if not isinstance(source_payload, dict):
        raise ValueError("FOL validation selection is invalid")
    values: set[str] = set()
    for band in ("low", "middle", "high"):
        ids = source_payload.get(band)
        if not isinstance(ids, list) or not all(isinstance(value, str) and value for value in ids):
            raise ValueError("FOL validation selection is invalid")
        values.update(ids)
    return values


def _records(root: Path, source: str) -> tuple[dict[str, BenchmarkExample], dict[str, OptimizationRecord]]:
    examples = {
        row.example_id: row
        for row in (BenchmarkExample.model_validate(row) for row in read_jsonl(root / "manifests" / f"controlled_{source}.jsonl"))
    }
    terminal = {
        row.sample_id: row
        for row in (OptimizationRecord.model_validate(row) for row in resolved_terminal_payloads(root, source).values())
        if row.checkpoint == 100 and row.status is RecordStatus.complete and row.state_path
    }
    if set(examples) != set(terminal):
        raise ValueError(f"FOL terminal O+ records are incomplete for {source}")
    return examples, terminal


def _acceptance_rates(root: Path) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[bool]] = {}
    for row in read_jsonl(root / "perturbation_semantic_outcomes.jsonl"):
        source, sample_id, accepted = row.get("source"), row.get("sample_id"), row.get("accepted")
        if isinstance(source, str) and isinstance(sample_id, str) and type(accepted) is bool:
            grouped.setdefault((source, sample_id), []).append(accepted)
    return {key: sum(values) / len(values) for key, values in grouped.items() if values}


def _objective(*, model: Any, tokenizer: Any, attack_text: str, config: Any) -> TransformerAttackObjective:
    token_ids = tokenizer(attack_text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    device = _embedding_device(model)
    token_ids = token_ids.to(device=device, dtype=torch.long)
    return TransformerAttackObjective(
        model,
        frozen_prompt_token_ids=token_ids,
        answer_token_ids=_anchor_token_ids(tokenizer, tuple(config.optimization.answer_anchors), device),
        refusal_token_ids=_anchor_token_ids(tokenizer, tuple(config.optimization.refusal_anchors), device),
        epsilon=config.optimization.epsilon,
        lambda_fol=config.optimization.lambda_fol,
        gamma_z=config.optimization.gamma_z,
        gamma_u=config.optimization.gamma_u,
    )


def _leaf(z: torch.Tensor, u: torch.Tensor, *, z0: torch.Tensor, u0: torch.Tensor) -> EditableState:
    return EditableState(
        z=z.detach().clone().requires_grad_(True),
        u=u.detach().clone().requires_grad_(True),
        z0=z0.detach().clone(),
        u0=u0.detach().clone(),
    )


def _curvature(
    objective: TransformerAttackObjective, *, z: torch.Tensor, u: torch.Tensor, sample_id: str, count: int
) -> float:
    values = []
    for direction_index in range(count):
        seed = int(canonical_hash({"sample": sample_id, "control": "curvature", "direction": direction_index})[:16], 16) % (2**31)
        state = _leaf(z, u, z0=z, u0=u)
        dz, du = random_joint_direction(tuple(z.shape), tuple(u.shape), torch.Generator(device="cpu").manual_seed(seed))
        dz, du = dz.to(device=z.device, dtype=z.dtype), du.to(device=u.device, dtype=u.dtype)
        hz, hu = objective.hvp(state, (dz, du))
        values.append(abs(float((hz * dz).sum().detach().cpu() + (hu * du).sum().detach().cpu())))
    return sum(values) / len(values)


def _roughness(
    objective: TransformerAttackObjective, *, z: torch.Tensor, u: torch.Tensor, sample_id: str, count: int, radius: float
) -> float:
    values = []
    for direction_index in range(count):
        seed = int(canonical_hash({"sample": sample_id, "control": "roughness", "direction": direction_index})[:16], 16) % (2**31)
        dz, du = random_joint_direction(tuple(z.shape), tuple(u.shape), torch.Generator(device="cpu").manual_seed(seed))
        state = _leaf(
            z + radius * dz.to(device=z.device, dtype=z.dtype),
            u + radius * du.to(device=u.device, dtype=u.dtype),
            z0=z,
            u0=u,
        )
        values.append(float(objective.evaluate(state, include_fol=False).attack_loss.detach().cpu()))
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _perplexity(example: BenchmarkExample, *, model: Any, tokenizer: Any) -> float:
    token_ids = tokenizer(example.attack_text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    embedding = model.get_input_embeddings().weight
    with torch.inference_mode():
        output = model(input_ids=token_ids.to(embedding.device), use_cache=False)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise ValueError("language model output has no logits")
    return causal_perplexity(logits, token_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", action="append")
    args = parser.parse_args()
    config = load_config(args.config)
    sources = _selected_sources(tuple(config.fol.sources), args.source)
    root = ROOT / config.run.output_root / "fol_boundary"
    model_path = config.models.surrogate.local_path
    if model_path is None:
        raise ValueError("FOL controls require a local model")
    acceptance = _acceptance_rates(root)
    handle = load_local_qwen(validate_model_assets(model_path))
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("local FOL control model did not load")
        device, dtype = handle.model.get_input_embeddings().weight.device, handle.model.get_input_embeddings().weight.dtype
        rows: list[dict[str, object]] = []
        for source in sources:
            examples, terminal = _records(root, source)
            validation_ids = _validation_ids(root, source)
            if len(validation_ids) != config.fol.validation_per_source:
                raise ValueError("FOL validation count drifted")
            for sample_id in sorted(validation_ids):
                example, record = examples[sample_id], terminal[sample_id]
                if not all(value is not None and math.isfinite(float(value)) for value in (record.fol, record.attack_loss, record.internal_margin)):
                    raise ValueError("FOL terminal diagnostics are incomplete")
                payload = torch.load(str(record.state_path), map_location="cpu", weights_only=True)
                if not isinstance(payload, dict) or not isinstance(payload.get("z"), torch.Tensor) or not isinstance(payload.get("u"), torch.Tensor):
                    raise ValueError("FOL terminal state is invalid")
                z, u = payload["z"].to(device=device, dtype=dtype), payload["u"].to(device=device, dtype=dtype)
                objective = _objective(model=handle.model, tokenizer=handle.tokenizer, attack_text=example.attack_text, config=config)
                rows.append({
                    "source": source,
                    "sample_id": sample_id,
                    "fol": float(record.fol),
                    "attack_loss": float(record.attack_loss),
                    "internal_margin": float(record.internal_margin),
                    "prompt_length": int(record.counters.prompt_tokens),
                    "perplexity": _perplexity(example, model=handle.model, tokenizer=handle.tokenizer),
                    "curvature": _curvature(objective, z=z, u=u, sample_id=sample_id, count=config.fol.hvp_directions),
                    "roughness": _roughness(
                        objective, z=z, u=u, sample_id=sample_id, count=8,
                        radius=config.fol.micro_noise_multiplier * config.optimization.epsilon,
                    ),
                    "semantic_acceptance_rate": acceptance.get((source, sample_id), 0.0),
                })
    finally:
        handle.close()
    written = _append_control_rows(root / "controls.jsonl", rows)
    print(json.dumps({"controls": len(rows), "written": written}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
