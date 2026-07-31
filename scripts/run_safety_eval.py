"""Bounded command line entrypoint for safety-evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import httpx

from benchmark.safety_eval.config import ExperimentConfig, load_config
from benchmark.safety_eval.execution import (
    ExecutionError,
    ExecutionMode,
    ExecutionRequest,
    TensorOptimizationSettings,
    build_local_qwen_tensor_executor,
    load_local_qwen,
    run_execution,
)
from benchmark.safety_eval.pipeline import materialize_records_from_disk
from benchmark.safety_eval.pipeline import generate_materialized_records, judge_response_records
from benchmark.safety_eval.io import read_jsonl
from benchmark.safety_eval.judging import OctopusLocalJudge, Qwen32CompatJudge, thresholds
from benchmark.safety_eval.runner import SerialTargetBarrier
from benchmark.safety_eval.result_aggregation import (
    write_judgment_summaries,
    write_materialization_summaries,
    write_paired_judgment_differences,
)
from benchmark.safety_eval.runtime import validate_model_assets
from benchmark.safety_eval.schema import MaterializationRecord, RecordStatus, ResponseRecord, JudgmentRecord
from benchmark.safety_eval.semantic import QwenHiddenMeanEncoder


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_EXECUTABLE_TENSOR_METHODS = frozenset(
    ("init", "random_mutation", "zol", "pez", "gbda", "gbda_official", "gcg", "jailbound_o_minus", "jailbound_o_plus", "dual_branch")
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    manifests = commands.add_parser("manifests", help="delegate controlled-manifest construction")
    _add_common_arguments(manifests)
    manifests.add_argument("--candidate-pool", type=_positive_int, default=200)

    validate = commands.add_parser("validate", help="report approved experiment identities")
    _add_common_arguments(validate)

    status = commands.add_parser("status", help="report lock and serial target markers")
    _add_common_arguments(status)

    smoke = commands.add_parser("run-smoke", help="run a bounded locked-manifest preflight or smoke attempt")
    _add_common_arguments(smoke)
    smoke.add_argument("--source", required=True)
    smoke.add_argument("--method", required=True)
    smoke.add_argument("--limit", type=_positive_int, default=1)
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--execute", action="store_true")

    optimize = commands.add_parser("optimize", help="execute one complete locked source-method optimization cell")
    _add_common_arguments(optimize)
    optimize.add_argument("--source", required=True)
    optimize.add_argument("--method", required=True)
    optimize.add_argument("--limit", type=_positive_int)

    materialize = commands.add_parser("materialize", help="project saved states using the frozen semantic threshold")
    _add_common_arguments(materialize)
    materialize.add_argument("--final-only", action="store_true")

    target = commands.add_parser("run-target", help="generate and judge one fully materialized target")
    _add_common_arguments(target)
    target.add_argument("--target", required=True)
    target.add_argument("--source", help="limit generation and judging to one configured source")
    target.add_argument("--method", action="append", help="limit generation and judging to configured methods")

    analyze = commands.add_parser("analyze", help="write count-first aggregate result tables")
    _add_common_arguments(analyze)
    return parser


def _resolve_output_root(config: ExperimentConfig, override: Path | None) -> Path:
    selected = override if override is not None else Path(config.run.output_root)
    return selected if selected.is_absolute() else REPOSITORY_ROOT / selected


def _validate_summary(config: ExperimentConfig) -> dict[str, object]:
    source_count = len(config.data.sources)
    return {
        "judge_keys": [config.judging.primary.key, config.judging.secondary.key],
        "method_ids": list(config.optimization.methods),
        "planned_sample_count": source_count * config.data.samples_per_source,
        "samples_per_source": config.data.samples_per_source,
        "schema_version": config.run.schema_version,
        "source_count": source_count,
        "target_keys": [target.key for target in config.models.targets],
    }


def _status_summary(config: ExperimentConfig, output_root: Path) -> dict[str, object]:
    locked = (output_root / config.run.locked_config_name).is_file()
    run_manifest = (output_root / "run_manifest.json").is_file()
    return {
        "locked_root": locked and run_manifest,
        "optimization_execution": "available",
        "serial_targets": [
            {
                "complete": (output_root / "responses" / target.key / "TARGET_COMPLETE.json").is_file(),
                "target": target.key,
            }
            for target in config.models.targets
        ],
    }


def _run_manifest_builder(config_path: Path, candidate_pool: int) -> int:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "build_safety_eval_manifests.py"),
        "--config",
        str(config_path),
        "--candidate-pool",
        str(candidate_pool),
    ]
    return subprocess.run(command, check=False).returncode


def _run_smoke(
    config: ExperimentConfig,
    output_root: Path,
    *,
    source: str,
    method: str,
    limit: int,
    dry_run: bool,
    execute: bool,
) -> dict[str, object]:
    if source not in config.data.sources:
        raise ValueError("source is not configured")
    if method not in config.optimization.methods:
        raise ValueError("method is not configured")
    if execute and not dry_run and method not in _EXECUTABLE_TENSOR_METHODS:
        raise ValueError("--execute supports only tensor methods: init, zol, jailbound_o_minus, jailbound_o_plus, dual_branch")
    model_path = config.models.surrogate.local_path
    if model_path is None:
        raise ValueError("surrogate local model path is required")
    mode = ExecutionMode.dry_run if dry_run else ExecutionMode.smoke
    request = ExecutionRequest(
        output_root=output_root,
        locked_config_name=config.run.locked_config_name,
        schema_version=config.run.schema_version,
        local_model_path=Path(model_path),
        source=source,
        method=method,
        checkpoints=(0,) if execute and not dry_run and method == "init" else tuple(config.optimization.checkpoints),
        requested_limit=limit,
        seed=config.run.seed,
    )
    if execute and not dry_run:
        tensor_settings = TensorOptimizationSettings(
            checkpoints=tuple(config.optimization.checkpoints),
            update_budget=config.optimization.update_budget,
            dual_branch_updates=dict(config.optimization.dual_branch_updates),
            candidate_cap=config.optimization.candidate_cap,
            prefix_tokens=config.optimization.prefix_tokens,
            editable_seed_tokens=config.optimization.editable_seed_tokens,
            learning_rate=config.optimization.learning_rate,
            lambda_fol=config.optimization.lambda_fol,
            epsilon=config.optimization.epsilon,
            gamma_z=config.optimization.gamma_z,
            gamma_u=config.optimization.gamma_u,
            grad_clip=config.optimization.grad_clip,
            answer_anchors=tuple(config.optimization.answer_anchors),
            refusal_anchors=tuple(config.optimization.refusal_anchors),
            gbda_learning_rate=getattr(config.optimization, "gbda_learning_rate", None),
            gcg_search_width=getattr(config.optimization, "gcg_search_width", 32),
        )
        summary = run_execution(
            request,
            mode=mode,
            model_loader=load_local_qwen,
            executor=build_local_qwen_tensor_executor(tensor_settings),
        )
    else:
        summary = run_execution(request, mode=mode)
    return {
        "completed_records": summary.completed_records,
        "failed_records": summary.failed_records,
        "method": method,
        "mode": mode.value,
        "selected_records": summary.selected_records,
        "source": source,
    }


def _semantic_threshold(config: ExperimentConfig, output_root: Path) -> float:
    configured = Path(config.semantic.threshold_artifact)
    candidates = [configured if configured.is_absolute() else REPOSITORY_ROOT / configured]
    candidates.append(output_root / "manifests" / "semantic_calibration.json")
    for path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        threshold = payload.get("threshold") if isinstance(payload, dict) else None
        if isinstance(threshold, (int, float)) and not isinstance(threshold, bool) and math.isfinite(threshold) and 0.0 <= threshold <= 1.0:
            return float(threshold)
        raise ValueError("semantic calibration artifact has an invalid threshold")
    raise ValueError("semantic calibration artifact is missing")


def _run_materialization(config: ExperimentConfig, output_root: Path, final_only: bool) -> dict[str, object]:
    threshold = _semantic_threshold(config, output_root)
    surrogate_path = config.models.surrogate.local_path
    semantic_path = config.models.semantic_encoder.local_path
    if surrogate_path is None or semantic_path is None:
        raise ValueError("materialization requires configured local surrogate and semantic encoder paths")
    handle = load_local_qwen(validate_model_assets(surrogate_path))
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("local surrogate failed to load")
        embedding = handle.model.get_input_embeddings().weight.detach()
        encoder = QwenHiddenMeanEncoder(
            semantic_path,
            tokenizer=handle.tokenizer,
            model=handle.model,
            revision=validate_model_assets(semantic_path).revision,
        )

        def similarity(before: str, after: str) -> float:
            vectors = encoder.encode([before, after])
            return float(vectors[0] @ vectors[1])

        summary = materialize_records_from_disk(
            output_root,
            vocabulary_embeddings=embedding,
            tokenizer=handle.tokenizer,
            semantic_similarity=similarity,
            semantic_threshold=threshold,
            final_only=final_only,
        )
    finally:
        handle.close()
    return {
        "selected_records": summary.selected_records,
        "written_records": summary.written_records,
        "failed_records": summary.failed_records,
        "final_only": final_only,
    }


def _configured_target(config: ExperimentConfig, target_key: str) -> Any:
    matches = [target for target in config.models.targets if target.key == target_key]
    if len(matches) != 1:
        raise ValueError("target is not configured exactly once")
    target = matches[0]
    if target.local_path is None:
        raise ValueError("target requires a local model path")
    return target


def _selected_sources(config: ExperimentConfig, source: str | None) -> tuple[str, ...]:
    if source is None:
        return tuple(config.data.sources)
    if source not in config.data.sources:
        raise ValueError("source is not configured")
    return (source,)


def _selected_methods(config: ExperimentConfig, methods: Sequence[str] | None) -> tuple[str, ...]:
    if methods is None:
        return tuple(config.optimization.methods)
    selected = tuple(methods)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("methods must be a non-empty unique configured list")
    if set(selected) - set(config.optimization.methods):
        raise ValueError("method is not configured")
    return selected


def _final_materializations(
    config: ExperimentConfig,
    output_root: Path,
    *,
    sources: tuple[str, ...],
    methods: tuple[str, ...],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for source in sources:
        manifest = read_jsonl(output_root / "manifests" / f"controlled_{source}.jsonl")
        sample_ids = {str(row.get("example_id")) for row in manifest}
        if len(sample_ids) != config.data.samples_per_source:
            raise ValueError(f"controlled manifest is incomplete for {source}")
        for method in methods:
            checkpoint = 0 if method == "init" else 100
            rows = [
                MaterializationRecord.model_validate(row)
                for row in read_jsonl(output_root / "optimization" / source / method / "materialization.jsonl")
            ]
            selected = [row for row in rows if row.checkpoint == checkpoint]
            by_sample = {row.sample_id: row for row in selected}
            if len(by_sample) != len(selected) or set(by_sample) != sample_ids:
                raise ValueError(f"final materializations are incomplete for {source}/{method}")
            records.extend(by_sample[sample_id].model_dump(mode="json") for sample_id in sorted(sample_ids))
    expected = len(sources) * len(methods) * config.data.samples_per_source
    if len(records) != expected:
        raise ValueError("final materialization count does not match the locked matrix")
    return records


def _responses_for_target(
    config: ExperimentConfig,
    output_root: Path,
    target_key: str,
    *,
    sources: tuple[str, ...],
    methods: tuple[str, ...],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for source in sources:
        for method in methods:
            checkpoint = 0 if method == "init" else 100
            rows = [
                ResponseRecord.model_validate(row)
                for row in read_jsonl(output_root / "responses" / target_key / source / method / "records.jsonl")
            ]
            selected = [row for row in rows if row.checkpoint == checkpoint]
            if len({row.sample_id for row in selected}) != config.data.samples_per_source:
                raise ValueError(f"responses are incomplete for {target_key}/{source}/{method}")
            records.extend(row.model_dump(mode="json") for row in sorted(selected, key=lambda row: row.sample_id))
    return records


def _judgment_count(
    config: ExperimentConfig,
    output_root: Path,
    *,
    judge_key: str,
    target_key: str,
    threshold: float,
    sources: tuple[str, ...],
    methods: tuple[str, ...],
) -> int:
    expected = len(sources) * len(methods) * config.data.samples_per_source
    keys: set[tuple[str, str, str, int]] = set()
    for source in sources:
        for method in methods:
            rows = [
                JudgmentRecord.model_validate(row)
                for row in read_jsonl(output_root / "judgments" / judge_key / target_key / source / method / "records.jsonl")
            ]
            for row in rows:
                if row.threshold != threshold:
                    continue
                if row.status not in {RecordStatus.complete, RecordStatus.failed}:
                    raise ValueError("judgment records must be terminal")
                key = (row.source, row.method, row.sample_id, row.checkpoint)
                if key in keys:
                    raise ValueError("duplicate terminal judgment key")
                keys.add(key)
    if len(keys) != expected:
        raise ValueError(f"judgments are incomplete for {judge_key} at threshold {threshold}")
    return len(keys)


def _compatibility_endpoint_ready(endpoint: str, model: str) -> None:
    response = httpx.get(f"{endpoint.rstrip('/')}/models", timeout=15.0)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data") if isinstance(payload, dict) else None
    ids = {item.get("id") for item in models if isinstance(item, dict)} if isinstance(models, list) else set()
    if model not in ids:
        raise ValueError("compatibility judge endpoint does not expose the configured model")


def _run_target(
    config: ExperimentConfig,
    output_root: Path,
    target_key: str,
    source: str | None = None,
    methods: tuple[str, ...] | None = None,
) -> dict[str, object]:
    target = _configured_target(config, target_key)
    barrier = SerialTargetBarrier(output_root, tuple(model.key for model in config.models.targets))
    sources = _selected_sources(config, source)
    selected_methods = _selected_methods(config, methods)
    if source is None and methods is None:
        barrier.require_ready(target_key)
    materializations = _final_materializations(
        config,
        output_root,
        sources=sources,
        methods=selected_methods,
    )
    target_resolved = validate_model_assets(target.local_path)
    target_handle = load_local_qwen(target_resolved)
    try:
        if target_handle.model is None or target_handle.tokenizer is None:
            raise ValueError("target model failed to load")
        generate_materialized_records(
            output_root,
            materializations,
            model=target_handle.model,
            tokenizer=target_handle.tokenizer,
            target_key=target_key,
            target_revision=target_resolved.revision,
            max_new_tokens=config.judging.max_new_tokens,
        )
    finally:
        target_handle.close()

    responses = _responses_for_target(
        config,
        output_root,
        target_key,
        sources=sources,
        methods=selected_methods,
    )
    primary = config.judging.primary
    octopus_path = config.models.octopus.local_path
    if octopus_path is None:
        raise ValueError("primary judge requires a local model path")
    octopus_resolved = validate_model_assets(octopus_path)
    octopus_handle = load_local_qwen(octopus_resolved)
    try:
        if octopus_handle.model is None or octopus_handle.tokenizer is None:
            raise ValueError("primary judge model failed to load")
        judge = OctopusLocalJudge(model=octopus_handle.model, tokenizer=octopus_handle.tokenizer, revision=octopus_resolved.revision)
        for threshold in thresholds(primary.threshold, primary.threshold_offsets):
            judge_response_records(output_root, responses, judge=judge, threshold=threshold)
    finally:
        octopus_handle.close()

    secondary = config.judging.secondary
    if secondary.endpoint is None or secondary.model is None:
        raise ValueError("secondary judge endpoint and model are required")
    _compatibility_endpoint_ready(secondary.endpoint, secondary.model)
    with Qwen32CompatJudge(
        endpoint=secondary.endpoint,
        model=secondary.model,
        max_new_tokens=config.judging.max_new_tokens,
    ) as judge:
        for threshold in thresholds(secondary.threshold, secondary.threshold_offsets):
            judge_response_records(output_root, responses, judge=judge, threshold=threshold)

    primary_counts = [
        _judgment_count(
            config,
            output_root,
            judge_key=primary.key,
            target_key=target_key,
            threshold=threshold,
            sources=sources,
            methods=selected_methods,
        )
        for threshold in thresholds(primary.threshold, primary.threshold_offsets)
    ]
    secondary_counts = [
        _judgment_count(
            config,
            output_root,
            judge_key=secondary.key,
            target_key=target_key,
            threshold=threshold,
            sources=sources,
            methods=selected_methods,
        )
        for threshold in thresholds(secondary.threshold, secondary.threshold_offsets)
    ]
    response_count = len(responses)
    base_primary = _judgment_count(
        config,
        output_root,
        judge_key=primary.key,
        target_key=target_key,
        threshold=primary.threshold,
        sources=sources,
        methods=selected_methods,
    )
    base_secondary = _judgment_count(
        config,
        output_root,
        judge_key=secondary.key,
        target_key=target_key,
        threshold=secondary.threshold,
        sources=sources,
        methods=selected_methods,
    )
    if any(count != response_count for count in (*primary_counts, *secondary_counts)):
        raise ValueError("not every threshold has one terminal judgment per response")
    if source is None and methods is None:
        barrier.mark_complete(
            target_key,
            response_count=response_count,
            primary_judgment_count=base_primary,
            secondary_judgment_count=base_secondary,
        )
    result: dict[str, object] = {
        "target": target_key,
        "response_count": response_count,
        "primary_judgment_count": base_primary,
        "secondary_judgment_count": base_secondary,
    }
    if source is not None:
        result["source"] = source
    if methods is not None:
        result["methods"] = list(selected_methods)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "manifests":
            if args.output_root is not None:
                raise ValueError("manifests does not support --output-root; configure it in the YAML")
            return _run_manifest_builder(args.config, args.candidate_pool)

        config = load_config(args.config)
        if args.command == "validate":
            print(json.dumps(_validate_summary(config), sort_keys=True))
            return 0

        output_root = _resolve_output_root(config, args.output_root)
        if args.command == "materialize":
            print(json.dumps(_run_materialization(config, output_root, args.final_only), sort_keys=True))
            return 0
        if args.command == "run-target":
            selected_methods = tuple(args.method) if args.method is not None else None
            if args.source is None and selected_methods is None:
                result = _run_target(config, output_root, args.target)
            elif selected_methods is None:
                result = _run_target(config, output_root, args.target, args.source)
            else:
                result = _run_target(config, output_root, args.target, args.source, selected_methods)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "analyze":
            sensitivity = write_judgment_summaries(output_root)
            paired = write_paired_judgment_differences(output_root)
            fidelity = write_materialization_summaries(output_root)
            print(json.dumps({
                "judge_sensitivity": str(sensitivity),
                "materialization_fidelity": str(fidelity),
                "paired_asr": str(paired),
            }, sort_keys=True))
            return 0
        if args.command in {"run-smoke", "optimize"}:
            if args.command == "optimize":
                summary = _run_smoke(
                    config,
                    output_root,
                    source=args.source,
                    method=args.method,
                    limit=args.limit or config.data.samples_per_source,
                    dry_run=False,
                    execute=True,
                )
                print(json.dumps(summary, sort_keys=True))
                return 0 if summary["failed_records"] == 0 else 1
            summary = _run_smoke(
                config,
                output_root,
                source=args.source,
                method=args.method,
                limit=args.limit,
                dry_run=args.dry_run,
                execute=args.execute,
            )
            print(json.dumps(summary, sort_keys=True))
            return 0 if args.dry_run or summary["failed_records"] == 0 else 1

        summary = _status_summary(config, output_root)
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary["locked_root"] else 1
    except (ExecutionError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
