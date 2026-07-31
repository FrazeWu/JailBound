"""Generate immutable H1-v2 baselines from optimized embeddings without text output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark.safety_eval.config import load_h1_v2_config
from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.generation import generate_embedding_response_record
from benchmark.safety_eval.io import JsonlLedger
from benchmark.safety_eval.runtime import validate_model_assets

from run_fol_embedding_generation import _full_embedding_input
from benchmark.safety_eval.h1_v2_runtime import h1_v2_eligible_records


ROOT = Path(__file__).resolve().parents[1]
METHOD = "fol_h1_v2"


def _select_shard_ids(sample_ids: tuple[str, ...], *, shard_index: int, shard_count: int) -> tuple[str, ...]:
    """Partition a fixed source-local baseline set without changing its identities."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("H1-v2 baseline shard settings are invalid")
    return tuple(sample_id for index, sample_id in enumerate(sorted(sample_ids)) if index % shard_count == shard_index)


def _state_payload(path: str | None) -> dict[str, torch.Tensor]:
    if not path:
        raise ValueError("H1-v2 optimized baseline has no editable state")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("z"), torch.Tensor) or not isinstance(payload.get("u"), torch.Tensor):
        raise ValueError("H1-v2 optimized baseline state is invalid")
    return {"z": payload["z"], "u": payload["u"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    config = load_h1_v2_config(args.config)
    sources = tuple(args.source or config.h1_v2.sources)
    if set(sources) - set(config.h1_v2.sources):
        raise ValueError("H1-v2 baseline generation requested an unconfigured source")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("H1-v2 baseline shard settings are invalid")
    root = ROOT / config.h1_v2.output_root
    try:
        payload = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        run_id, config_hash = payload["run_id"], payload["config_hash"]
    except (KeyError, OSError, json.JSONDecodeError) as error:
        raise ValueError("H1-v2 run manifest is invalid") from error
    model_path = config.base.models.surrogate.local_path
    if model_path is None:
        raise ValueError("H1-v2 baseline generation requires a local target")
    resolved = validate_model_assets(model_path)
    handle = load_local_qwen(resolved, attention_backend=config.base.run.attention_implementation)
    written = failed = selected = 0
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("H1-v2 local target did not load")
        for source in sources:
            examples, optimized = h1_v2_eligible_records(root, source)
            if len(examples) < config.h1_v2.minimum_baseline_safe_count:
                raise ValueError("H1-v2 baseline has too few eligible candidates")
            ledger = JsonlLedger(
                root / "responses" / config.base.models.targets[0].key / source / METHOD / "records.jsonl",
                key_fields=("sample_id", "checkpoint"),
            )
            for sample_id in _select_shard_ids(
                tuple(examples), shard_index=args.shard_index, shard_count=args.shard_count,
            ):
                selected += 1
                if ledger.contains_key({"sample_id": sample_id, "checkpoint": 0}):
                    continue
                inputs_embeds, attention_mask = _full_embedding_input(
                    handle.model,
                    handle.tokenizer,
                    attack_text=examples[sample_id].attack_text,
                    state_payload=_state_payload(optimized[sample_id].state_path),
                )
                record = generate_embedding_response_record(
                    model=handle.model,
                    tokenizer=handle.tokenizer,
                    schema_version=config.base.run.schema_version,
                    run_id=run_id,
                    config_hash=config_hash,
                    sample_id=sample_id,
                    source=source,
                    method=METHOD,
                    checkpoint=0,
                    target_key=config.base.models.targets[0].key,
                    target_revision=resolved.revision,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=config.base.judging.max_new_tokens,
                )
                if ledger.append_once(record.model_dump(mode="json")):
                    written += 1
                    failed += int(record.status.value == "failed")
    finally:
        handle.close()
    print(json.dumps({"failed": failed, "selected": selected, "written": written}, sort_keys=True))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
