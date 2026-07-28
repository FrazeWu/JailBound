"""Freeze FOL validation bands and radius-calibration IDs from O+ terminal records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.fol_records import resolved_terminal_payloads
from benchmark.safety_eval.fol_runtime import causal_perplexity, select_fol_experiment
from benchmark.safety_eval.io import atomic_write_json, atomic_write_jsonl, read_jsonl
from benchmark.safety_eval.manifest import FolCandidate
from benchmark.safety_eval.runtime import validate_model_assets
from benchmark.safety_eval.schema import BenchmarkExample, OptimizationRecord, RecordStatus


ROOT = Path(__file__).resolve().parents[1]


def _input_ids(encoded: object) -> torch.Tensor:
    if isinstance(encoded, dict):
        value = encoded.get("input_ids")
    else:
        value = getattr(encoded, "input_ids", None)
    if not isinstance(value, torch.Tensor):
        raise ValueError("tokenizer did not return tensor input IDs")
    return value


def _perplexity(example: BenchmarkExample, *, model: Any, tokenizer: Any) -> float:
    token_ids = _input_ids(tokenizer(example.attack_text, return_tensors="pt", add_special_tokens=True))
    embedding = model.get_input_embeddings().weight
    with torch.inference_mode():
        output = model(input_ids=token_ids.to(embedding.device), use_cache=False)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise ValueError("language model output has no logits tensor")
    return causal_perplexity(logits, token_ids)


def _candidates(root: Path, source: str, *, model: Any, tokenizer: Any) -> tuple[FolCandidate, ...]:
    examples = {
        row.example_id: row
        for row in (
            BenchmarkExample.model_validate(row)
            for row in read_jsonl(root / "manifests" / f"controlled_{source}.jsonl")
        )
    }
    final = {
        row.sample_id: row
        for row in (
            OptimizationRecord.model_validate(row)
            for row in resolved_terminal_payloads(root, source).values()
        )
        if row.checkpoint == 100 and row.status is RecordStatus.complete
    }
    if set(final) != set(examples):
        raise ValueError(f"FOL O+ terminal records are incomplete for {source}")
    values = []
    for sample_id in sorted(examples):
        example, record = examples[sample_id], final[sample_id]
        if not all(
            value is not None and math.isfinite(float(value))
            for value in (record.fol, record.attack_loss, record.internal_margin)
        ):
            raise ValueError(f"FOL O+ terminal diagnostics are incomplete for {source}")
        values.append(FolCandidate(
            sample_id=sample_id,
            source=source,
            fol=float(record.fol),
            risk_category=example.risk_category,
            initial_label=bool(float(record.internal_margin) >= 0.0),
            attack_loss=float(record.attack_loss),
            token_length=record.counters.prompt_tokens,
            perplexity=_perplexity(example, model=model, tokenizer=tokenizer),
        ))
    return tuple(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = ROOT / config.run.output_root / "fol_boundary"
    artifact = root / "manifests" / "validation_selection.json"
    if artifact.is_file():
        print(json.dumps({"artifact": str(artifact), "status": "existing"}, sort_keys=True))
        return 0
    model_path = config.models.surrogate.local_path
    if model_path is None:
        raise ValueError("FOL validation selection requires a local surrogate")
    handle = load_local_qwen(validate_model_assets(model_path))
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("local FOL surrogate did not load")
        selections = {}
        for source in config.fol.sources:
            selection = select_fol_experiment(
                _candidates(root, source, model=handle.model, tokenizer=handle.tokenizer),
                seed=config.run.seed,
            )
            if selection.status != "ready":
                raise ValueError(f"FOL validation matching is inconclusive for {source}")
            selections[source] = {
                "low": list(selection.low),
                "middle": list(selection.middle),
                "high": list(selection.high),
                "radius_calibration": list(selection.radius_calibration),
                "matching": {
                    "caliper": selection.matching_caliper,
                    "pair_distances": list(selection.matching_distances),
                },
            }
    finally:
        handle.close()
    payload = {
        "config_hash": json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))["config_hash"],
        "sources": selections,
    }
    atomic_write_json(artifact, payload)
    for source, selection in selections.items():
        rows = [
            {"sample_id": sample_id, "band": band}
            for band in ("low", "middle", "high", "radius_calibration")
            for sample_id in selection[band]
        ]
        path = root / "manifests" / f"fol_boundary_{source}.jsonl"
        atomic_write_jsonl(path, rows)
    print(json.dumps({"artifact": str(artifact), "status": "frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
