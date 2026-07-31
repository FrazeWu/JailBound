"""Freeze H1-v2 baseline-safe endpoints from numeric diagnostics and primary labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from benchmark.safety_eval.config import load_h1_v2_config
from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.fol_runtime import ConfirmatoryFolCandidate, select_h1_v2_validation
from benchmark.safety_eval.io import atomic_write_json, canonical_hash, read_jsonl
from benchmark.safety_eval.runtime import validate_model_assets
from benchmark.safety_eval.schema import RecordStatus
from benchmark.safety_eval.fol_runtime import causal_perplexity
from benchmark.safety_eval.h1_v2_runtime import h1_v2_eligible_records


ROOT = Path(__file__).resolve().parents[1]


def _perplexity(model: object, tokenizer: object, text: str) -> float:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if not isinstance(token_ids, torch.Tensor):
        raise ValueError("H1-v2 candidate tokenization is invalid")
    embeddings = model.get_input_embeddings().weight
    with torch.no_grad():
        logits = model(input_ids=token_ids.to(embeddings.device)).logits
    return causal_perplexity(logits, token_ids)


def _exploratory_ids(root: Path, source: str) -> set[str]:
    return {
        str(row["example_id"])
        for row in read_jsonl(root / "fol_boundary" / "manifests" / f"controlled_{source}.jsonl")
        if isinstance(row.get("example_id"), str)
    }


def _balance_diagnostics(
    candidates: list[ConfirmatoryFolCandidate], selection: object
) -> dict[str, object]:
    """Summarize numeric endpoint balance without retaining sample content."""
    candidate_by_id = {row.sample_id: row for row in candidates}
    low_ids = tuple(getattr(selection, "low"))
    high_ids = tuple(getattr(selection, "high"))
    diagnostics: dict[str, object] = {}
    for name in ("attack_loss", "token_length", "perplexity"):
        low_values = [float(getattr(candidate_by_id[sample_id], name)) for sample_id in low_ids]
        high_values = [float(getattr(candidate_by_id[sample_id], name)) for sample_id in high_ids]
        combined = low_values + high_values
        mean = sum(combined) / len(combined)
        variance = sum((value - mean) ** 2 for value in combined) / len(combined)
        scale = math.sqrt(variance)
        diagnostics[name] = {
            "low_mean": sum(low_values) / len(low_values),
            "high_mean": sum(high_values) / len(high_values),
            "standardized_mean_difference": (
                (sum(high_values) / len(high_values) - sum(low_values) / len(low_values)) / scale
                if scale > 0.0 else 0.0
            ),
        }
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_h1_v2_config(args.config)
    root = ROOT / config.h1_v2.output_root
    artifact = root / "manifests" / "h1_v2_validation_selection.json"
    if artifact.exists():
        print(json.dumps({"artifact": str(artifact), "status": "existing"}, sort_keys=True))
        return 0
    labels = {(str(row.get("source")), str(row.get("sample_id"))): row for row in read_jsonl(root / "baseline_labels.jsonl")}
    model_path = config.base.models.surrogate.local_path
    if model_path is None:
        raise ValueError("H1-v2 validation selection requires a local surrogate")
    handle = load_local_qwen(validate_model_assets(model_path), attention_backend=config.base.run.attention_implementation)
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("H1-v2 local surrogate did not load")
        sources = {}
        for source in config.h1_v2.sources:
            examples, terminal = h1_v2_eligible_records(root, source)
            if len(examples) < config.h1_v2.minimum_baseline_safe_count:
                raise ValueError(f"H1-v2 has too few eligible candidates for {source}")
            source_labels = {sample_id: labels.get((source, sample_id)) for sample_id in examples}
            if any(row is None or type(row.get("baseline_safe")) is not bool for row in source_labels.values()):
                raise ValueError(f"H1-v2 baseline labels are incomplete for {source}")
            candidates = []
            for sample_id in sorted(examples):
                record = terminal[sample_id]
                if record.status is not RecordStatus.complete or not all(
                    value is not None and math.isfinite(float(value))
                    for value in (record.fol, record.attack_loss)
                ):
                    raise ValueError(f"H1-v2 terminal diagnostics are invalid for {source}")
                if source_labels[sample_id]["baseline_safe"]:
                    candidates.append(ConfirmatoryFolCandidate(
                        sample_id=sample_id, source=source, fol=float(record.fol),
                        risk_category=examples[sample_id].risk_category, attack_loss=float(record.attack_loss),
                        token_length=record.counters.prompt_tokens,
                        perplexity=_perplexity(handle.model, handle.tokenizer, examples[sample_id].attack_text),
                        baseline_safe=True,
                    ))
            selection = select_h1_v2_validation(
                tuple(candidates), exploratory_ids=_exploratory_ids(ROOT / config.base.run.output_root, source), seed=config.base.run.seed,
            )
            if selection.status != "ready":
                raise ValueError(f"H1-v2 selection is inconclusive for {source}")
            fol_by_id = {row.sample_id: row.fol for row in candidates}
            sources[source] = {
                "low": list(selection.low), "middle": list(selection.middle), "high": list(selection.high),
                "reserves": list(selection.reserves), "fol_by_id": fol_by_id,
                "matching": {
                    "mode": selection.matching_mode,
                    "caliper": selection.matching_caliper,
                    "pair_distances": list(selection.matching_distances),
                    "risk_category_exact": selection.risk_category_matching,
                    "balance_diagnostics": _balance_diagnostics(candidates, selection),
                },
            }
    finally:
        handle.close()
    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    payload = {"config_hash": run_manifest["config_hash"], "sources": sources}
    payload["selected_ids_hash"] = canonical_hash(sorted(
        (source, sample_id) for source, values in sources.items()
        for band in ("low", "middle", "high") for sample_id in values[band]
    ))
    atomic_write_json(artifact, payload)
    print(json.dumps({"artifact": str(artifact), "status": "frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
