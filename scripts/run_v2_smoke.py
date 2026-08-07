#!/usr/bin/env python3
"""Run the isolated reviewer_eval.v2 one-sample smoke pipeline."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import httpx


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from benchmark.safety_eval.config import load_v2_config
from benchmark.safety_eval.execution import (
    ExecutionMode,
    ExecutionRequest,
    TensorOptimizationSettings,
    build_local_qwen_tensor_executor,
    load_local_qwen,
    run_execution,
)
from benchmark.safety_eval.io import read_jsonl
from benchmark.safety_eval.judging import OctopusLocalJudge, Qwen32CompatJudge
from benchmark.safety_eval.pipeline import generate_v2_materialized_records_from_local_assets, judge_response_records
from benchmark.safety_eval.result_aggregation import (
    load_judgment_rows,
    write_judgment_summaries,
    write_materialization_summaries,
    write_paired_judgment_differences,
)
from benchmark.safety_eval.runtime import PreflightError, validate_model_assets, validate_v2_provenance_ledgers
from benchmark.safety_eval.schema import V2MaterializationRecord, V2ResponseRecord
from benchmark.safety_eval.v2_pipeline import materialize_v2_terminal_records

from build_safety_eval_v2_manifests import build_manifests


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=int, default=1)
    parser.add_argument("--launch-secondary-vllm", action="store_true")
    parser.add_argument("--secondary-startup-timeout", type=float, default=300.0)
    return parser.parse_args(argv)


def _rooted(config_path: Path, value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def _settings(config: Any, *, tokenizer_revision: str) -> TensorOptimizationSettings:
    optimization = config.optimization
    return TensorOptimizationSettings(
        checkpoints=tuple(optimization.checkpoints),
        update_budget=optimization.update_budget,
        dual_branch_updates=dict(optimization.dual_branch_updates),
        candidate_cap=optimization.candidate_cap,
        prefix_tokens=optimization.prefix_tokens,
        learning_rate=optimization.learning_rate,
        gbda_learning_rate=optimization.gbda_learning_rate,
        gcg_search_width=optimization.gcg_search_width,
        lambda_fol=optimization.lambda_fol,
        epsilon=optimization.epsilon,
        gamma_z=optimization.gamma_z,
        gamma_u=optimization.gamma_u,
        grad_clip=optimization.grad_clip,
        answer_anchors=tuple(optimization.answer_anchors),
        refusal_anchors=tuple(optimization.refusal_anchors),
        prefix_token_text=optimization.prefix_initialization.token_text,
        tokenizer_revision=tokenizer_revision,
    )


def _optimize(config: Any, root: Path) -> None:
    source, method = config.data.sources[0], config.optimization.methods[0]
    model_path = _rooted(Path(), config.models.surrogate.local_path)
    resolved = validate_model_assets(model_path)
    request = ExecutionRequest(
        output_root=root,
        locked_config_name=config.run.locked_config_name,
        schema_version=config.run.schema_version,
        local_model_path=model_path,
        source=source,
        method=method,
        checkpoints=tuple(config.optimization.checkpoints),
        requested_limit=1,
        seed=config.run.seed,
    )
    summary = run_execution(
        request,
        mode=ExecutionMode.smoke,
        model_loader=lambda resolved: load_local_qwen(
            resolved, attention_backend=config.run.attention_implementation
        ),
        executor=build_local_qwen_tensor_executor(_settings(config, tokenizer_revision=resolved.tokenizer_hash)),
    )
    if summary.failed_records:
        raise RuntimeError("v2 smoke optimization wrote failure records")


def _materialize(config: Any, root: Path) -> list[dict[str, object]]:
    source, method = config.data.sources[0], config.optimization.methods[0]
    resolved = validate_model_assets(_rooted(Path(), config.models.surrogate.local_path))
    loaded = load_local_qwen(resolved, attention_backend=config.run.attention_implementation)
    try:
        if loaded.model is None or loaded.tokenizer is None:
            raise RuntimeError("surrogate loader returned an empty handle")
        embedding = loaded.model.get_input_embeddings().weight.detach()
        records = materialize_v2_terminal_records(
            root, source=source, method=method,
            vocabulary_embeddings=embedding, tokenizer=loaded.tokenizer,
            surrogate_tokenizer_sha256=resolved.tokenizer_hash,
        )
        return [record.model_dump(mode="json") for record in records]
    finally:
        loaded.close()


def _generate(config: Any, root: Path, materializations: list[dict[str, object]]) -> list[dict[str, object]]:
    target = config.models.targets[0]
    summary = generate_v2_materialized_records_from_local_assets(
        root, materializations,
        target_model_path=_rooted(Path(), target.local_path),
        target_key=target.key,
        attention_backend=config.run.attention_implementation,
        max_new_tokens=config.judging.max_new_tokens,
    )
    if summary.failed_records:
        raise RuntimeError("v2 smoke target generation wrote failure records")
    source, method = config.data.sources[0], config.optimization.methods[0]
    resolved = validate_model_assets(_rooted(Path(), target.local_path))
    current_materializations = {
        V2MaterializationRecord.model_validate(row).materialization_sha256
        for row in materializations
    }
    responses_by_materialization: dict[str, dict[str, object]] = {}
    for row in read_jsonl(root / "responses" / target.key / source / method / "records.jsonl"):
        response = V2ResponseRecord.model_validate(row)
        if (
            response.materialization_sha256 in current_materializations
            and response.target_revision == resolved.revision
            and response.target_tokenizer_sha256 == resolved.tokenizer_hash
        ):
            if response.materialization_sha256 in responses_by_materialization:
                raise RuntimeError("v2 smoke target response ledger duplicates a current materialization")
            responses_by_materialization[response.materialization_sha256] = response.model_dump(mode="json")
    if set(responses_by_materialization) != current_materializations:
        raise RuntimeError("v2 smoke target response ledger lacks the current materializations")
    return list(responses_by_materialization.values())


def _judge_primary(config: Any, root: Path, responses: list[dict[str, object]]) -> None:
    resolved = validate_model_assets(_rooted(Path(), config.models.octopus.local_path))
    loaded = load_local_qwen(resolved, attention_backend=config.run.attention_implementation)
    try:
        if loaded.model is None or loaded.tokenizer is None:
            raise RuntimeError("primary judge loader returned an empty handle")
        judge = OctopusLocalJudge(model=loaded.model, tokenizer=loaded.tokenizer, revision=resolved.revision)
        for offset in config.judging.primary.threshold_offsets:
            summary = judge_response_records(root, responses, judge=judge, threshold=config.judging.primary.threshold + offset)
            if summary.failed_records:
                raise RuntimeError("v2 smoke primary judge wrote failure records")
    finally:
        loaded.close()


def _judge_secondary(config: Any, root: Path, responses: list[dict[str, object]]) -> None:
    secondary = config.judging.secondary
    secondary_path = getattr(secondary, "local_path", None)
    if secondary_path is None:
        raise ValueError("v2 secondary judge requires a local snapshot")
    resolved = validate_model_assets(_rooted(Path(), secondary_path))
    with Qwen32CompatJudge(
        endpoint=secondary.endpoint,
        model=secondary.model,
        revision=resolved.revision,
    ) as judge:
        for offset in secondary.threshold_offsets:
            summary = judge_response_records(root, responses, judge=judge, threshold=secondary.threshold + offset)
            if summary.failed_records:
                raise RuntimeError("v2 smoke secondary judge wrote failure records")


@contextmanager
def _secondary_judge_server(config: Any, root: Path, *, startup_timeout: float):
    """Start the secondary vLLM only after all GPU-local stages have released memory."""
    if startup_timeout <= 0:
        raise ValueError("secondary startup timeout must be positive")
    secondary_path = getattr(config.judging.secondary, "local_path", None)
    if secondary_path is None:
        raise ValueError("v2 secondary judge requires a local snapshot")
    secondary_path = _rooted(Path(), secondary_path)
    endpoint = config.judging.secondary.endpoint.rstrip("/")
    if not endpoint.startswith("http://127.0.0.1:"):
        raise ValueError("automatic secondary vLLM launch requires a localhost endpoint")
    port = endpoint.rsplit(":", 1)[-1].removesuffix("/v1")
    if not port.isdigit():
        raise ValueError("secondary endpoint must include a numeric port")
    log_path = root / "logs" / "secondary_vllm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(secondary_path), "--host", "127.0.0.1", "--port", port,
        "--served-model-name", config.judging.secondary.model,
        "--dtype", "half", "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096", "--enforce-eager",
    ]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        deadline = time.monotonic() + startup_timeout
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("secondary vLLM exited before becoming ready")
                try:
                    response = httpx.get(f"{endpoint}/models", timeout=2.0)
                    models = response.json().get("data", [])
                    if response.is_success and any(row.get("id") == config.judging.secondary.model for row in models):
                        yield
                        return
                except (httpx.HTTPError, ValueError, TypeError):
                    pass
                time.sleep(1.0)
            raise RuntimeError("secondary vLLM did not become ready before timeout")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30.0)


_PAIRED_ASR_FIELDS = (
    "Judge", "Target", "Source", "Method", "Threshold", "Denominator",
    "Net ASR change", "Net ASR change (%)", "Method-only", "Init-only",
    "McNemar p", "Schema", "Branch", "Transport", "Judge revision", "Target revision",
    "Target tokenizer SHA256", "Materialization SHA256s", "Provenance",
)


def _write_analysis(_: Any, root: Path) -> None:
    write_materialization_summaries(root)
    write_judgment_summaries(root)
    if any(row.get("method") == "init" for row in load_judgment_rows(root)):
        write_paired_judgment_differences(root)
        return
    destination = root / "analysis" / "paired_asr.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=_PAIRED_ASR_FIELDS).writeheader()


def _reject_legacy_v2_provenance(root: Path) -> None:
    """Reject incomplete pre-integrity artifacts before any stage can resume."""
    try:
        validate_v2_provenance_ledgers(root)
    except PreflightError as error:
        raise ValueError("legacy v2 provenance is invalid") from error


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_v2_config(args.config)
    if not config.run.smoke_mode:
        raise ValueError("run_v2_smoke requires run.smoke_mode: true")
    root = _rooted(args.config, config.run.output_root)
    _reject_legacy_v2_provenance(root)
    build_manifests(args.config, candidate_pool=args.candidate_pool, sources=config.data.sources)
    _optimize(config, root)
    materializations = _materialize(config, root)
    responses = _generate(config, root, materializations)
    _judge_primary(config, root, responses)
    server = (
        _secondary_judge_server(config, root, startup_timeout=args.secondary_startup_timeout)
        if args.launch_secondary_vllm else nullcontext()
    )
    with server:
        _judge_secondary(config, root, responses)
    _write_analysis(config, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
