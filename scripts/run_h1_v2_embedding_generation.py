"""Batch-generate H1-v2's frozen, semantically accepted embedding perturbations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from benchmark.safety_eval.config import load_h1_v2_config
from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.generation import embedding_state_hash, generate_from_embedding_batch
from benchmark.safety_eval.io import JsonlLedger, read_jsonl
from benchmark.safety_eval.runtime import validate_model_assets
from benchmark.safety_eval.schema import FailureKind, RecordStatus, ResponseRecord
from benchmark.safety_eval.h1_v2_runtime import h1_v2_eligible_records

from run_fol_embedding_generation import _full_embedding_input
from run_fol_generation import _perturbed_payload


ROOT = Path(__file__).resolve().parents[1]
METHOD = "fol_h1_v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1 or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("H1-v2 batch and shard settings are invalid")
    config = load_h1_v2_config(args.config)
    if args.source not in config.h1_v2.sources:
        raise ValueError("H1-v2 embedding generation requested an unconfigured source")
    root = ROOT / config.h1_v2.output_root
    try:
        run = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        run_id, config_hash = run["run_id"], run["config_hash"]
    except (KeyError, OSError, json.JSONDecodeError) as error:
        raise ValueError("H1-v2 run manifest is invalid") from error
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in read_jsonl(root / "selected_directions.jsonl"):
        if row.get("source") != args.source:
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("H1-v2 selected direction has an invalid sample identity")
        grouped[sample_id].append(dict(row, kind="perturbation"))
    if not grouped:
        raise ValueError("H1-v2 selected direction ledger is empty for the requested source")
    sample_ids = tuple(sorted(grouped))
    sample_ids = tuple(sample_id for index, sample_id in enumerate(sample_ids) if index % args.shard_count == args.shard_index)
    model_path = config.base.models.surrogate.local_path
    if model_path is None:
        raise ValueError("H1-v2 embedding generation requires a local target")
    resolved = validate_model_assets(model_path)
    handle = load_local_qwen(resolved, attention_backend=config.base.run.attention_implementation)
    ledger = JsonlLedger(
        root / "responses" / config.base.models.targets[0].key / args.source / METHOD / "records.jsonl",
        key_fields=("sample_id", "checkpoint"),
    )
    written = failed = 0
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("H1-v2 local target did not load")
        examples, optimized = h1_v2_eligible_records(root, args.source)
        embeddings = handle.model.get_input_embeddings().weight.detach()
        for source_sample_id in sample_ids:
            pending = [
                row for row in sorted(grouped[source_sample_id], key=lambda item: int(item["direction_index"]))
                if not ledger.contains_key({"sample_id": row["perturbation_id"], "checkpoint": 0})
            ]
            for start in range(0, len(pending), args.batch_size):
                chunk = pending[start : start + args.batch_size]
                states = [
                    _full_embedding_input(
                        handle.model, handle.tokenizer, attack_text=examples[source_sample_id].attack_text,
                        state_payload=_perturbed_payload(optimized[source_sample_id], row, embeddings=embeddings),
                    )
                    for row in chunk
                ]
                inputs = torch.cat([state[0] for state in states], dim=0)
                masks = torch.cat([state[1] for state in states], dim=0)
                try:
                    generated = generate_from_embedding_batch(
                        handle.model, handle.tokenizer, inputs_embeds=inputs, attention_mask=masks,
                        max_new_tokens=config.base.judging.max_new_tokens,
                    )
                    errors = [None] * len(chunk)
                except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
                    generated = [None] * len(chunk)
                    errors = [error] * len(chunk)
                for row, state, result, error in zip(chunk, states, generated, errors, strict=True):
                    perturbation_id = row["perturbation_id"]
                    if not isinstance(perturbation_id, str):
                        raise ValueError("H1-v2 perturbation identity is invalid")
                    common = {
                        "schema_version": config.base.run.schema_version, "run_id": run_id, "config_hash": config_hash,
                        "sample_id": perturbation_id, "source": args.source, "method": METHOD, "checkpoint": 0,
                        "target_key": config.base.models.targets[0].key, "target_revision": resolved.revision,
                        "prompt_hash": embedding_state_hash(state[0]),
                    }
                    if result is None:
                        record = ResponseRecord(**common, response="", input_tokens=0, generated_tokens=0,
                            status=RecordStatus.failed, failure_kind=FailureKind.generation,
                            failure_reason=f"H1-v2 embedding generation error: {type(error).__name__}")
                    else:
                        record = ResponseRecord(**common, response=result.response, input_tokens=result.input_tokens,
                            generated_tokens=result.generated_tokens, status=RecordStatus.complete,
                            failure_kind=None, failure_reason=None)
                    if ledger.append_once(record.model_dump(mode="json")):
                        written += 1
                        failed += int(record.status is RecordStatus.failed)
    finally:
        handle.close()
    print(json.dumps({"failed": failed, "written": written}, sort_keys=True))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
