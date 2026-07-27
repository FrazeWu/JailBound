"""Execute the registered O+ trajectories for the frozen FOL candidate manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.reviewer_eval.config import load_config
from benchmark.reviewer_eval.execution import (
    ExecutionMode,
    ExecutionRequest,
    TensorOptimizationSettings,
    build_local_qwen_tensor_executor,
    load_local_qwen,
    run_execution,
)


ROOT = Path(__file__).resolve().parents[1]


def _settings(config: object) -> TensorOptimizationSettings:
    optimization = config.optimization
    return TensorOptimizationSettings(
        checkpoints=tuple(optimization.checkpoints),
        update_budget=optimization.update_budget,
        dual_branch_updates=dict(optimization.dual_branch_updates),
        candidate_cap=optimization.candidate_cap,
        prefix_tokens=optimization.prefix_tokens,
        editable_seed_tokens=optimization.editable_seed_tokens,
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
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", action="append")
    args = parser.parse_args()
    config = load_config(args.config)
    sources = tuple(args.source or config.fol.sources)
    if set(sources) - set(config.fol.sources):
        raise ValueError("FOL optimization requested an unconfigured source")
    model_path = config.models.surrogate.local_path
    if model_path is None:
        raise ValueError("FOL optimization requires the local surrogate")
    fol_root = ROOT / config.run.output_root / "fol_boundary"
    settings = _settings(config)
    results = []
    for source in sources:
        request = ExecutionRequest(
            output_root=fol_root,
            locked_config_name=config.run.locked_config_name,
            schema_version=config.run.schema_version,
            local_model_path=model_path,
            source=source,
            method="jailbound_o_plus",
            checkpoints=tuple(config.optimization.checkpoints),
            requested_limit=45,
            seed=config.run.seed,
        )
        summary = run_execution(
            request,
            mode=ExecutionMode.smoke,
            model_loader=load_local_qwen,
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
