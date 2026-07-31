"""Resume direct-embedding FOL generation in equal-length per-sample batches."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.generation import embedding_state_hash, generate_from_embedding_batch
from benchmark.safety_eval.io import JsonlLedger
from benchmark.safety_eval.runtime import validate_model_assets
from benchmark.safety_eval.schema import FailureKind, RecordStatus, ResponseRecord
from benchmark.safety_eval.fol_runtime import select_id_shard

from run_fol_embedding_generation import ROOT, _full_embedding_input, _resolve_root
from run_fol_generation import _accepted_metadata, _final_records, _perturbed_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--input-fol-root", type=Path)
    parser.add_argument("--output-fol-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    config = load_config(args.config)
    if args.source not in config.fol.sources:
        raise ValueError("FOL embedding batch resume requested an unconfigured source")
    input_root = _resolve_root(args.input_fol_root or (Path(config.run.output_root) / "fol_boundary"))
    output_root = _resolve_root(args.output_fol_root or input_root.with_name(f"{input_root.name}_inputs_embeds"))
    source_manifest = json.loads((input_root / "run_manifest.json").read_text(encoding="utf-8"))
    run_id, config_hash = source_manifest.get("run_id"), source_manifest.get("config_hash")
    if not isinstance(run_id, str) or not isinstance(config_hash, str):
        raise ValueError("source FOL run manifest is invalid")
    metadata = [
        row for row in _accepted_metadata(
            input_root, sources=(args.source,), directions_per_radius=config.fol.directions_per_radius
        )
        if row["source"] == args.source
    ]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in metadata:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError("FOL batch metadata is invalid")
        grouped[sample_id].append(row)
    response_ledger = JsonlLedger(
        output_root / "responses" / config.models.targets[0].key / args.source / "fol_boundary" / "records.jsonl",
        key_fields=("sample_id", "checkpoint"),
    )
    audit_ledger = JsonlLedger(output_root / "embedding_state_audit.jsonl", key_fields=("perturbation_id",))
    model_path = config.models.surrogate.local_path
    if model_path is None:
        raise ValueError("FOL embedding batch resume requires a local target")
    resolved = validate_model_assets(model_path)
    handle = load_local_qwen(resolved, attention_backend=config.run.attention_implementation)
    written = failed = 0
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("local FOL target did not load")
        examples, optimized = _final_records(input_root, args.source)
        embedding_weight = handle.model.get_input_embeddings().weight.detach()
        for source_sample_id in select_id_shard(
            tuple(grouped), shard_index=args.shard_index, shard_count=args.shard_count
        ):
            pending = [
                row for row in grouped[source_sample_id]
                if not response_ledger.contains_key({"sample_id": row["perturbation_id"], "checkpoint": 0})
            ]
            for start in range(0, len(pending), args.batch_size):
                chunk = pending[start : start + args.batch_size]
                states = [
                    _full_embedding_input(
                        handle.model,
                        handle.tokenizer,
                        attack_text=examples[source_sample_id].attack_text,
                        state_payload=_perturbed_payload(optimized[source_sample_id], row, embeddings=embedding_weight),
                    )
                    for row in chunk
                ]
                inputs = torch.cat([item[0] for item in states], dim=0)
                masks = torch.cat([item[1] for item in states], dim=0)
                try:
                    results = generate_from_embedding_batch(
                        handle.model, handle.tokenizer, inputs_embeds=inputs,
                        attention_mask=masks, max_new_tokens=config.judging.max_new_tokens,
                    )
                    errors = [None] * len(chunk)
                except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
                    results = [None] * len(chunk)
                    errors = [error] * len(chunk)
                for row, state, result, error in zip(chunk, states, results, errors, strict=True):
                    perturbation_id = row["perturbation_id"]
                    if not isinstance(perturbation_id, str):
                        raise ValueError("FOL batch perturbation ID is invalid")
                    state_hash = embedding_state_hash(state[0])
                    audit_ledger.append_once({
                        "perturbation_id": perturbation_id,
                        "source": args.source,
                        "sample_id": source_sample_id,
                        "kind": row["kind"],
                        "state_hash": state_hash,
                        "input_tokens": int(state[0].shape[1]),
                        "embedding_l2_norm": float(torch.linalg.vector_norm(state[0].float()).detach().cpu()),
                    })
                    common = dict(
                        schema_version=config.run.schema_version, run_id=run_id, config_hash=config_hash,
                        sample_id=perturbation_id, source=args.source, method="fol_boundary", checkpoint=0,
                        target_key=config.models.targets[0].key, target_revision=resolved.revision, prompt_hash=state_hash,
                    )
                    if result is None:
                        record = ResponseRecord(**common, response="", input_tokens=0, generated_tokens=0,
                            status=RecordStatus.failed, failure_kind=FailureKind.generation,
                            failure_reason=f"target embedding batch generation error: {type(error).__name__}")
                    else:
                        record = ResponseRecord(**common, response=result.response, input_tokens=result.input_tokens,
                            generated_tokens=result.generated_tokens, status=RecordStatus.complete,
                            failure_kind=None, failure_reason=None)
                    if response_ledger.append_once(record.model_dump(mode="json")):
                        written += 1
                        failed += int(record.status is RecordStatus.failed)
    finally:
        handle.close()
    print(json.dumps({"failed": failed, "written": written}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
