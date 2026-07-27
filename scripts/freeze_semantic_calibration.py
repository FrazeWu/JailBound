"""Freeze the held-out semantic threshold without persisting calibration text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch

from benchmark.reviewer_eval.config import ExperimentConfig, load_config
from benchmark.reviewer_eval.datasets import load_source_with_report
from benchmark.reviewer_eval.execution import load_local_qwen
from benchmark.reviewer_eval.io import atomic_write_json, read_jsonl
from benchmark.reviewer_eval.manifest import map_raw_candidates
from benchmark.reviewer_eval.runtime import validate_model_assets
from benchmark.reviewer_eval.semantic import (
    CalibrationCandidate,
    QwenHiddenMeanEncoder,
    build_calibration_pairs,
    choose_canonical_label,
    freeze_semantic_calibration,
    load_taxonomy_mapping,
    select_calibration_holdouts,
)


ROOT = Path(__file__).resolve().parents[1]
_ENTITY = re.compile(r"\b(?:[A-Z][A-Za-z0-9_-]{1,}|\d[A-Za-z0-9_-]*)\b")


def _output_root(config: ExperimentConfig, override: Path | None) -> Path:
    selected = override if override is not None else Path(config.run.output_root)
    return selected if selected.is_absolute() else ROOT / selected


def _entities_preserved(before: str, after: str) -> bool:
    required = {item.casefold() for item in _ENTITY.findall(before)}
    observed = after.casefold()
    return all(entity in observed for entity in required)


def _model_device(model: Any) -> torch.device:
    weight = model.get_input_embeddings().weight
    return weight.device


def _paraphrase(model: Any, tokenizer: Any, text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Rewrite the user request in English while preserving its intent and named entities. Return only the rewritten request.",
        },
        {"role": "user", "content": text},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(_model_device(model))
    with torch.inference_mode():
        output_ids = model.generate(input_ids, max_new_tokens=256, do_sample=False)
    generated = output_ids[0][input_ids.shape[-1] :]
    return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()


def _candidates(
    config: ExperimentConfig,
    *,
    encoder: QwenHiddenMeanEncoder,
    mapping: dict[str, Any],
    label_embeddings: dict[str, Any],
) -> dict[str, tuple[CalibrationCandidate, ...]]:
    source_hashes = {source: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for source, path in config.data.paths.items()}
    output: dict[str, tuple[CalibrationCandidate, ...]] = {}
    for source in config.data.sources:
        raw, _ = load_source_with_report(
            source,
            ROOT / config.data.paths[source],
            ROOT / config.data.harmbench_targets_path if source == "harmbench" else None,
        )
        chosen = sorted(
            raw,
            key=lambda row: hashlib.sha256(f"{config.run.seed}|{row.source_row_id}".encode()).hexdigest(),
        )[:200]
        mapped = map_raw_candidates(
            chosen,
            mapping=mapping,
            label_embeddings=label_embeddings,
            encoder=encoder,
            source_file=str(config.data.paths[source]),
            source_sha256=source_hashes[source],
            seed=config.run.seed,
        )
        output[source] = tuple(
            CalibrationCandidate(
                example_id=row.example_id,
                source=row.source,
                risk_category=row.risk_category,
                intent=row.intent,
            )
            for row in mapped
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = _output_root(config, args.output_root)
    artifact_path = output_root / "manifests" / "semantic_calibration.json"
    if artifact_path.is_file():
        print(json.dumps({"artifact": str(artifact_path), "status": "existing"}, sort_keys=True))
        return 0
    model_path = config.models.semantic_encoder.local_path
    if model_path is None:
        raise ValueError("semantic encoder requires a local model path")
    resolved = validate_model_assets(model_path)
    handle = load_local_qwen(resolved)
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("semantic model failed to load")
        encoder = QwenHiddenMeanEncoder(
            model_path,
            tokenizer=handle.tokenizer,
            model=handle.model,
            revision=resolved.revision,
        )
        mapping = load_taxonomy_mapping(ROOT / "configs/benchmark/reviewer_taxonomy_map.yaml")
        labels = list(mapping["risk_categories"]) + list(mapping["threat_domains"])
        descriptions = [
            mapping["risk_categories"].get(label, mapping["threat_domains"].get(label))["description"]
            for label in labels
        ]
        label_embeddings = dict(zip(labels, encoder.encode(descriptions), strict=True))
        candidates = _candidates(config, encoder=encoder, mapping=mapping, label_embeddings=label_embeddings)
        holdouts = []
        for source in config.data.sources:
            controlled_ids = frozenset(
                str(row["example_id"])
                for row in read_jsonl(output_root / "manifests" / f"controlled_{source}.jsonl")
            )
            holdouts.extend(
                select_calibration_holdouts(
                    candidates[source],
                    controlled_ids=controlled_ids,
                    per_source=config.semantic.calibration_examples_per_source,
                    seed=config.run.seed,
                )
            )

        risk_categories = list(mapping["risk_categories"])

        def category_for_text(text: str) -> str:
            label, _ = choose_canonical_label(text, risk_categories, label_embeddings, encoder)
            return label

        def similarity(left: str, right: str) -> float:
            vectors = encoder.encode([left, right])
            return float(vectors[0] @ vectors[1])

        pairs = build_calibration_pairs(
            tuple(holdouts),
            paraphrase=lambda text: _paraphrase(handle.model, handle.tokenizer, text),
            category_for_text=category_for_text,
            entities_preserved=_entities_preserved,
            similarity=similarity,
            max_attempts=config.semantic.max_mutation_attempts,
        )
    finally:
        handle.close()
    expected = 2 * len(config.data.sources) * config.semantic.calibration_examples_per_source
    if len(pairs) != expected:
        raise ValueError("semantic calibration did not retain every registered held-out pair")
    artifact = freeze_semantic_calibration(
        pairs,
        target_recall=config.semantic.target_positive_recall,
        encoder_revision=resolved.revision,
    )
    artifact.update({"config_hash": json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))["config_hash"]})
    atomic_write_json(artifact_path, artifact)
    print(json.dumps({"artifact": str(artifact_path), "pair_count": artifact["pair_count"], "status": "frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
