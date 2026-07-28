"""Generate FOL responses directly from continuous embeddings without projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.fol_runtime import select_id_shard
from benchmark.safety_eval.generation import generate_embedding_response_record
from benchmark.safety_eval.io import JsonlLedger, atomic_write_json
from benchmark.safety_eval.runtime import validate_model_assets

from run_fol_generation import _accepted_metadata, _final_records, _perturbed_payload


ROOT = Path(__file__).resolve().parents[1]


def _resolve_root(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def _input_ids(tokenizer: Any, attack_text: str) -> torch.Tensor:
    encoded = tokenizer(attack_text, return_tensors="pt", add_special_tokens=True)
    token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError("FOL frozen prompt tokenization did not produce one token sequence")
    return token_ids


def _full_embedding_input(
    model: Any,
    tokenizer: Any,
    *,
    attack_text: str,
    state_payload: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the optimizer layout ``[z | frozen prompt | u]`` exactly."""
    embedding_layer = model.get_input_embeddings()
    embedding_weight = getattr(embedding_layer, "weight", None)
    if not isinstance(embedding_weight, torch.Tensor):
        raise ValueError("FOL target has no tensor input embedding matrix")
    token_ids = _input_ids(tokenizer, attack_text).to(device=embedding_weight.device)
    with torch.no_grad():
        frozen_prompt = embedding_layer(token_ids).detach()
    z, u = state_payload["z"], state_payload["u"]
    if any(value.ndim != 3 or value.shape[0] != 1 for value in (z, u)):
        raise ValueError("FOL editable state must contain one rank-3 z/u pair")
    z = z.to(device=frozen_prompt.device, dtype=frozen_prompt.dtype)
    u = u.to(device=frozen_prompt.device, dtype=frozen_prompt.dtype)
    if z.shape[-1] != frozen_prompt.shape[-1] or u.shape[-1] != frozen_prompt.shape[-1]:
        raise ValueError("FOL editable state hidden size differs from target embedding size")
    inputs_embeds = torch.cat((z, frozen_prompt, u), dim=1).detach()
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    return inputs_embeds, attention_mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-fol-root", type=Path)
    parser.add_argument("--output-fol-root", type=Path)
    parser.add_argument("--source", action="append")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    sources = tuple(args.source or config.fol.sources)
    if set(sources) - set(config.fol.sources):
        raise ValueError("FOL embedding generation requested an unconfigured source")
    input_root = _resolve_root(
        args.input_fol_root or (Path(config.run.output_root) / "fol_boundary")
    )
    output_root = _resolve_root(
        args.output_fol_root or input_root.with_name(f"{input_root.name}_inputs_embeds")
    )
    if input_root.resolve() == output_root.resolve():
        raise ValueError("direct embedding FOL output root must differ from the source FOL root")

    metadata = _accepted_metadata(
        input_root,
        sources=sources,
        directions_per_radius=config.fol.directions_per_radius,
    )
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("FOL embedding shard index is invalid")
    selected_ids = set(select_id_shard(
        tuple(str(row["perturbation_id"]) for row in metadata),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    ))
    metadata = tuple(row for row in metadata if row["perturbation_id"] in selected_ids)
    if args.max_records is not None:
        if args.max_records < 1:
            raise ValueError("FOL embedding max records must be positive")
        metadata = metadata[:args.max_records]
    metadata_ledger = JsonlLedger(output_root / "selected_perturbations.jsonl", key_fields=("perturbation_id",))
    for row in metadata:
        metadata_ledger.append_once(row)

    source_manifest = json.loads((input_root / "run_manifest.json").read_text(encoding="utf-8"))
    run_id, config_hash = source_manifest.get("run_id"), source_manifest.get("config_hash")
    if not isinstance(run_id, str) or not isinstance(config_hash, str):
        raise ValueError("source FOL run manifest is invalid")
    atomic_write_json(output_root / "embedding_generation_provenance.json", {
        "generation_mode": "direct_inputs_embeds",
        "source_fol_root": str(input_root),
        "run_id": run_id,
        "config_hash": config_hash,
        "sources": sorted(config.fol.sources),
    })

    model_path = config.models.surrogate.local_path
    if model_path is None:
        raise ValueError("FOL embedding generation requires a local target")
    resolved = validate_model_assets(model_path)
    handle = load_local_qwen(resolved, attention_backend=config.run.attention_implementation)
    written = failed = 0
    state_hashes: set[str] = set()
    baseline_count = 0
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("local FOL target did not load")
        embedding_weight = handle.model.get_input_embeddings().weight.detach()
        source_records = {source: _final_records(input_root, source) for source in sources}
        response_ledgers: dict[str, JsonlLedger] = {}
        audit_ledger = JsonlLedger(output_root / "embedding_state_audit.jsonl", key_fields=("perturbation_id",))
        for row in metadata:
            source = row["source"]
            sample_id = row["sample_id"]
            perturbation_id = row["perturbation_id"]
            if not all(isinstance(value, str) and value for value in (source, sample_id, perturbation_id)):
                raise ValueError("FOL embedding generation metadata is invalid")
            example, optimization = source_records[source][0][sample_id], source_records[source][1][sample_id]
            inputs_embeds, attention_mask = _full_embedding_input(
                handle.model,
                handle.tokenizer,
                attack_text=example.attack_text,
                state_payload=_perturbed_payload(optimization, row, embeddings=embedding_weight),
            )
            record = generate_embedding_response_record(
                model=handle.model,
                tokenizer=handle.tokenizer,
                schema_version=config.run.schema_version,
                run_id=run_id,
                config_hash=config_hash,
                sample_id=perturbation_id,
                source=source,
                method="fol_boundary",
                checkpoint=0,
                target_key=config.models.targets[0].key,
                target_revision=resolved.revision,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=config.judging.max_new_tokens,
            )
            state_hashes.add(record.prompt_hash)
            if row["kind"] == "baseline":
                baseline_count += 1
            audit_ledger.append_once({
                "perturbation_id": perturbation_id,
                "source": source,
                "sample_id": sample_id,
                "kind": row["kind"],
                "state_hash": record.prompt_hash,
                "input_tokens": record.input_tokens,
                "embedding_l2_norm": float(torch.linalg.vector_norm(inputs_embeds.float()).detach().cpu()),
            })
            ledger = response_ledgers.setdefault(
                source,
                JsonlLedger(
                    output_root / "responses" / config.models.targets[0].key / source / "fol_boundary" / "records.jsonl",
                    key_fields=("sample_id", "checkpoint"),
                ),
            )
            if ledger.append_once(record.model_dump(mode="json")):
                written += 1
                failed += int(record.status.value == "failed")
    finally:
        handle.close()
    if len(state_hashes) <= baseline_count:
        raise RuntimeError("direct embedding FOL quality gate failed: perturbations did not produce distinct states")
    print(json.dumps({
        "baseline_states": baseline_count,
        "failed": failed,
        "selected": len(metadata),
        "unique_state_hashes": len(state_hashes),
        "written": written,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
