"""Execute the registered O+ trajectories for the frozen FOL candidate manifests."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.execution import (
    ExecutionMode,
    ExecutionRequest,
    TensorOptimizationSettings,
    build_local_qwen_tensor_executor,
    load_local_qwen,
    run_execution,
)


ROOT = Path(__file__).resolve().parents[1]


def _two_gpu_recovery_device_map(model_path: Path) -> dict[str, int | str]:
    try:
        layer_count = json.loads((model_path / "config.json").read_text(encoding="utf-8"))["num_hidden_layers"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("local recovery model has no valid layer count") from error
    if not isinstance(layer_count, int) or layer_count < 2:
        raise ValueError("local recovery model has an invalid layer count")
    split = layer_count // 2
    cpu_layers = {split - 1, layer_count - 1}
    return {
        "model.embed_tokens": 0,
        **{
            f"model.layers.{index}": "cpu" if index in cpu_layers else (0 if index < split else 1)
            for index in range(layer_count)
        },
        "model.norm": 1,
        "model.rotary_emb": 1,
        "lm_head": 1,
    }


def _settings(config: object, *, finite_difference_fol: bool = False) -> TensorOptimizationSettings:
    optimization = config.optimization
    return TensorOptimizationSettings(
        checkpoints=tuple(optimization.checkpoints),
        update_budget=optimization.update_budget,
        dual_branch_updates=dict(optimization.dual_branch_updates),
        candidate_cap=optimization.candidate_cap,
        prefix_tokens=optimization.prefix_tokens,
        prefix_token_text=getattr(
            getattr(optimization, "prefix_initialization", None), "token_text", "!"
        ),
        learning_rate=optimization.learning_rate,
        lambda_fol=optimization.lambda_fol,
        epsilon=optimization.epsilon,
        gamma_z=optimization.gamma_z,
        gamma_u=optimization.gamma_u,
        grad_clip=optimization.grad_clip,
        answer_anchors=tuple(optimization.answer_anchors),
        refusal_anchors=tuple(optimization.refusal_anchors),
        gbda_learning_rate=optimization.gbda_learning_rate,
        gcg_search_width=optimization.gcg_search_width,
        finite_difference_fol=finite_difference_fol,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--method", default="jailbound_o_plus")
    parser.add_argument("--sample-id-file", type=Path)
    parser.add_argument("--attention-backend", choices=("eager", "sdpa"), default="eager")
    args = parser.parse_args()
    config = load_config(args.config)
    sources = tuple(args.source or config.fol.sources)
    if set(sources) - set(config.fol.sources):
        raise ValueError("FOL optimization requested an unconfigured source")
    requested_sample_ids: tuple[str, ...] | None = None
    if args.sample_id_file is not None:
        if len(sources) != 1:
            raise ValueError("sample-ID recovery requires exactly one source")
        try:
            payload = json.loads(args.sample_id_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("sample-ID recovery file is invalid") from error
        if not isinstance(payload, list) or not payload or not all(isinstance(value, str) and value for value in payload):
            raise ValueError("sample-ID recovery file must be a non-empty string list")
        requested_sample_ids = tuple(payload)
    if args.attention_backend == "sdpa" and (
        args.method not in {
            "jailbound_o_plus_recovery_sdpa",
            "jailbound_o_plus_recovery_fd_sdpa",
        }
        or requested_sample_ids is None
    ):
        raise ValueError("SDPA is restricted to explicit jailbound_o_plus recovery sample IDs")
    memory_recovery = args.method in {
        "jailbound_o_plus_recovery_checkpointed",
        "jailbound_o_plus_recovery_rebalanced",
    }
    finite_difference_recovery = args.method in {
        "jailbound_o_plus_recovery_fd",
        "jailbound_o_plus_recovery_fd_sdpa",
    }
    if (memory_recovery or finite_difference_recovery) and requested_sample_ids is None:
        raise ValueError("restricted recovery is limited to explicit O+ recovery sample IDs")
    model_path = config.models.surrogate.local_path
    if model_path is None:
        raise ValueError("FOL optimization requires the local surrogate")
    fol_root = ROOT / config.run.output_root / "fol_boundary"
    settings = _settings(config, finite_difference_fol=finite_difference_recovery)
    results = []
    for source in sources:
        request = ExecutionRequest(
            output_root=fol_root,
            locked_config_name=config.run.locked_config_name,
            schema_version=config.run.schema_version,
            local_model_path=model_path,
            source=source,
            method=args.method,
            checkpoints=tuple(config.optimization.checkpoints),
            requested_limit=45,
            seed=config.run.seed,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            requested_sample_ids=requested_sample_ids,
        )
        summary = run_execution(
            request,
            mode=ExecutionMode.smoke,
            model_loader=partial(
                load_local_qwen,
                attention_backend=args.attention_backend,
                activation_checkpointing=memory_recovery,
                device_map=_two_gpu_recovery_device_map(model_path) if memory_recovery else "auto",
            ),
            executor=build_local_qwen_tensor_executor(settings),
        )
        results.append({
            "source": source,
            "selected_records": summary.selected_records,
            "completed_records": summary.completed_records,
            "failed_records": summary.failed_records,
        })
    print(json.dumps({"results": results}, sort_keys=True))
    return 0 if all(row["failed_records"] == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
