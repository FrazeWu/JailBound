"""Optimize the frozen H1-v2 candidate manifests with the registered O+ path."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

from benchmark.safety_eval.config import load_h1_v2_config
from benchmark.safety_eval.execution import ExecutionMode, ExecutionRequest, build_local_qwen_tensor_executor, load_local_qwen, run_execution
from benchmark.safety_eval.io import read_jsonl
from benchmark.safety_eval.fol_runtime import h1_v2_recovery_spec, select_h1_v2_pending_recovery_ids

from run_fol_candidate_optimization import _settings, _two_gpu_recovery_device_map


ROOT = Path(__file__).resolve().parents[1]


def _cpu_offload_recovery_device_map(model_path: Path) -> dict[str, int | str]:
    """Place the middle half of transformer layers on CPU for OOM recovery only."""
    try:
        layer_count = json.loads((model_path / "config.json").read_text(encoding="utf-8"))["num_hidden_layers"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("local recovery model has no valid layer count") from error
    if not isinstance(layer_count, int) or layer_count < 4:
        raise ValueError("local recovery model has an invalid layer count")
    edge_layers = layer_count // 4
    if edge_layers < 1:
        raise ValueError("local recovery model cannot retain GPU edge layers")
    return {
        "model.embed_tokens": 0,
        **{
            f"model.layers.{index}": 0 if index < edge_layers or index >= layer_count - edge_layers else "cpu"
            for index in range(layer_count)
        },
        "model.norm": 0,
        "model.rotary_emb": 0,
        "lm_head": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--recover-primary-oom", action="store_true")
    parser.add_argument(
        "--recovery-mode",
        choices=("eager", "checkpointed", "sdpa", "cpu_offload", "two_gpu", "two_gpu_checkpointed"),
        default="eager",
    )
    args = parser.parse_args()
    config = load_h1_v2_config(args.config)
    sources = tuple(args.source or config.h1_v2.sources)
    if set(sources) - set(config.h1_v2.sources):
        raise ValueError("H1-v2 optimization requested an unconfigured source")
    model_path = config.base.models.surrogate.local_path
    if model_path is None:
        raise ValueError("H1-v2 optimization requires a local surrogate")
    root = ROOT / config.h1_v2.output_root
    recovery_ids: tuple[str, ...] | None = None
    method = "jailbound_o_plus"
    activation_checkpointing = False
    attention_backend = config.base.run.attention_implementation
    device_map: str | dict[str, int | str] = "auto"
    if args.recover_primary_oom:
        if len(sources) != 1:
            raise ValueError("H1-v2 OOM recovery requires exactly one source")
        source = sources[0]
        primary = root / "optimization" / source / "jailbound_o_plus" / "records.jsonl"
        recovery_records = [
            record
            for path in (root / "optimization" / source).glob("jailbound_o_plus_recovery*/records.jsonl")
            for record in read_jsonl(path)
        ]
        recovery_ids = select_h1_v2_pending_recovery_ids(read_jsonl(primary), recovery_records)
        if not recovery_ids:
            raise ValueError("H1-v2 has no pending all-checkpoint primary OOM failures to recover")
        method, activation_checkpointing = h1_v2_recovery_spec(args.recovery_mode)
        if args.recovery_mode == "sdpa":
            attention_backend = "sdpa"
        if args.recovery_mode == "cpu_offload":
            device_map = _cpu_offload_recovery_device_map(model_path)
        if args.recovery_mode in {"two_gpu", "two_gpu_checkpointed"}:
            device_map = _two_gpu_recovery_device_map(model_path)
    summaries = []
    for source in sources:
        summary = run_execution(
            ExecutionRequest(
                output_root=root,
                locked_config_name=config.base.run.locked_config_name,
                schema_version=config.base.run.schema_version,
                local_model_path=model_path,
                source=source,
                method=method,
                checkpoints=tuple(config.base.optimization.checkpoints),
                requested_limit=config.h1_v2.candidate_count,
                seed=config.base.run.seed,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                requested_sample_ids=recovery_ids,
            ),
            mode=ExecutionMode.smoke,
            model_loader=partial(
                load_local_qwen,
                attention_backend=attention_backend,
                activation_checkpointing=activation_checkpointing,
                device_map=device_map,
            ),
            executor=build_local_qwen_tensor_executor(_settings(config.base)),
        )
        summaries.append({
            "source": source,
            "selected": summary.selected_records,
            "completed": summary.completed_records,
            "failed": summary.failed_records,
        })
    print(json.dumps({"results": summaries}, sort_keys=True))
    return 0 if all(item["failed"] == 0 for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
