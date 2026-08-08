"""Judge continuous/materialized responses with the frozen matched judges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple, Sequence

from omegaconf import OmegaConf

from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.io import canonical_hash, read_jsonl
from benchmark.safety_eval.judging import OctopusLocalJudge, Qwen32CompatJudge
from benchmark.safety_eval.materialization_ablation import MaterializationPair, canonical_pair_key, index_pairs
from benchmark.safety_eval.materialization_judging import (
    MATCHED_JUDGE_THRESHOLDS,
    PairedJudgmentRecord,
    judge_materialization_pair,
    merge_paired_judgments,
    paired_judgment_key,
)
from benchmark.safety_eval.runtime import validate_model_assets


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_KEY = "octopus_seval_14b"
SECONDARY_KEY = "qwen32_compat"


class JudgeSpec(NamedTuple):
    key: str
    threshold: float
    local_path: Path | None
    repo_id: str | None
    endpoint: str | None
    model: str | None


class JudgingConfig(NamedTuple):
    source_root: Path
    output_root: Path
    target_key: str
    target_revision: str
    primary: JudgeSpec
    secondary: JudgeSpec
    payload: dict[str, object]


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a probability")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a probability")
    return number


def _absolute(value: object, *, name: str) -> Path:
    path = Path(_text(value, name=name))
    return path if path.is_absolute() else ROOT / path


def _judge_spec(payload: dict[str, object], *, primary: bool) -> JudgeSpec:
    key = _text(payload.get("key"), name="judge key")
    threshold = _probability(payload.get("threshold"), name=f"{key} threshold")
    local_path = _absolute(payload.get("local_path"), name=f"{key} local_path") if primary else None
    repo_id = _text(payload.get("repo_id"), name=f"{key} repo_id") if primary else None
    endpoint = _text(payload.get("endpoint"), name=f"{key} endpoint") if not primary else None
    model = _text(payload.get("model"), name=f"{key} model") if not primary else None
    return JudgeSpec(key, threshold, local_path, repo_id, endpoint, model)


def load_judging_config(path: Path) -> JudgingConfig:
    """Load the ablation config and prove judge settings match the source lock."""
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    payload = _mapping(loaded, name="materialization config")
    source_root = _absolute(payload.get("source_root"), name="source_root")
    output_root = _absolute(payload.get("output_root"), name="output_root")
    model = _mapping(payload.get("model"), name="model")
    target_key = _text(model.get("key"), name="target key")
    target_revision = _text(model.get("revision"), name="target revision")
    judges = _mapping(payload.get("judges"), name="judges")
    primary = _judge_spec(_mapping(judges.get("primary"), name="primary judge"), primary=True)
    secondary = _judge_spec(_mapping(judges.get("secondary"), name="secondary judge"), primary=False)
    if primary.key != PRIMARY_KEY or secondary.key != SECONDARY_KEY:
        raise ValueError("materialization judging requires the two frozen matched judge keys")
    if {
        primary.key: primary.threshold,
        secondary.key: secondary.threshold,
    } != MATCHED_JUDGE_THRESHOLDS:
        raise ValueError("materialization judging requires the frozen matched judge thresholds")

    locked = _mapping(
        json.loads((source_root / "locked_config.json").read_text(encoding="utf-8")),
        name="matched locked config",
    )
    locked_judging = _mapping(locked.get("judging"), name="matched judging")
    locked_primary = _mapping(locked_judging.get("primary"), name="matched primary judge")
    locked_secondary = _mapping(locked_judging.get("secondary"), name="matched secondary judge")
    locked_octopus = _mapping(
        _mapping(locked.get("models"), name="matched models").get("octopus"),
        name="matched Octopus model",
    )
    matches = (
        locked_primary.get("key") == primary.key
        and locked_primary.get("threshold") == primary.threshold
        and locked_secondary.get("key") == secondary.key
        and locked_secondary.get("threshold") == secondary.threshold
        and locked_secondary.get("endpoint") == secondary.endpoint
        and locked_secondary.get("model") == secondary.model
        and locked_octopus.get("key") == primary.key
        and locked_octopus.get("repo_id") == primary.repo_id
        and Path(str(locked_octopus.get("local_path"))).resolve() == primary.local_path.resolve()
    )
    if not matches:
        raise ValueError("judge settings differ from the matched locked config")
    return JudgingConfig(
        source_root,
        output_root,
        target_key,
        target_revision,
        primary,
        secondary,
        payload,
    )


def _pairs(path: Path) -> tuple[MaterializationPair, ...]:
    pairs = tuple(
        MaterializationPair.model_validate(row)
        for row in read_jsonl(path)
    )
    index_pairs(pairs)
    return pairs


def _existing(path: Path) -> dict[tuple[str, str, object, int, str], PairedJudgmentRecord]:
    indexed: dict[tuple[str, str, object, int, str], PairedJudgmentRecord] = {}
    for row in read_jsonl(path):
        record = PairedJudgmentRecord.model_validate(row)
        key = paired_judgment_key(record)
        if key in indexed:
            raise ValueError(f"duplicate paired judgment: {key}")
        indexed[key] = record
    return indexed


def _selection(
    pairs: tuple[MaterializationPair, ...],
    existing: dict[tuple[str, str, object, int, str], PairedJudgmentRecord],
    *,
    judge_key: str,
    max_pairs: int | None,
) -> tuple[tuple[MaterializationPair, ...], dict[str, int | str]]:
    complete = tuple(pair for pair in pairs if pair.status == "complete")
    existing_for_judge = [record for key, record in existing.items() if key[-1] == judge_key]
    pending = tuple(
        pair
        for pair in complete
        if (record := existing.get((*canonical_pair_key(pair), judge_key))) is None or record.status == "failed"
    )
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max-pairs must be positive")
        pending = pending[:max_pairs]
    report: dict[str, int | str] = {
        "complete_generation_pairs": len(complete),
        "existing_complete_judgments": sum(record.status == "complete" for record in existing_for_judge),
        "existing_failed_judgments": sum(record.status == "failed" for record in existing_for_judge),
        "failed_generation_pairs": len(pairs) - len(complete),
        "judge_key": judge_key,
        "pending_pairs": len(pending),
    }
    return pending, report


def run_with_judge(
    config: JudgingConfig,
    pairs_path: Path,
    output_path: Path,
    *,
    judge: object,
    max_pairs: int | None = None,
) -> dict[str, int | str]:
    """Judge pending pairs and atomically merge terminal records."""
    judge_key = str(getattr(judge, "key", ""))
    spec = config.primary if judge_key == config.primary.key else config.secondary if judge_key == config.secondary.key else None
    if spec is None:
        raise ValueError(f"judge is not part of the frozen matched configuration: {judge_key}")
    pairs = _pairs(pairs_path)
    if any(pair.model_key != config.target_key or pair.model_revision != config.target_revision for pair in pairs):
        raise ValueError("materialization pair model provenance differs from the judging config")
    existing = _existing(output_path)
    config_hash = canonical_hash(config.payload)
    judge_revision = str(getattr(judge, "revision", ""))
    for key, record in existing.items():
        if key[-1] != judge_key:
            continue
        if (
            record.config_hash != config_hash
            or record.target_key != config.target_key
            or record.target_revision != config.target_revision
            or record.judge_revision != judge_revision
            or record.threshold != spec.threshold
        ):
            raise ValueError(f"existing paired judgment provenance differs from this run: {key}")
    pending, report = _selection(
        pairs,
        existing,
        judge_key=judge_key,
        max_pairs=max_pairs,
    )
    records = tuple(
        judge_materialization_pair(
            pair,
            judge=judge,
            run_id=f"materialization:{config_hash[:20]}",
            config_hash=config_hash,
            target_key=config.target_key,
            target_revision=config.target_revision,
            threshold=spec.threshold,
        )
        for pair in pending
    )
    appended, replaced = merge_paired_judgments(output_path, records)
    report.update({
        "written": appended + replaced,
        "complete_written": sum(record.status == "complete" for record in records),
        "failed_written": sum(record.status == "failed" for record in records),
    })
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--judge", choices=(PRIMARY_KEY, SECONDARY_KEY), required=True)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--runtime-endpoint",
        help="Transport-only endpoint override for the frozen secondary judge",
    )
    parser.add_argument(
        "--runtime-model-path",
        type=Path,
        help="Storage-only path override for the frozen primary judge",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.runtime_endpoint is not None and args.judge != SECONDARY_KEY:
        raise ValueError("runtime-endpoint is only valid for the secondary judge")
    if args.runtime_model_path is not None and args.judge != PRIMARY_KEY:
        raise ValueError("runtime-model-path is only valid for the primary judge")
    config = load_judging_config(args.config)
    pairs_path = args.pairs or config.output_root / "materialization_pairs.jsonl"
    output_path = args.output or config.output_root / "paired_judgments.jsonl"
    pairs = _pairs(pairs_path)
    _, report = _selection(
        pairs,
        _existing(output_path),
        judge_key=args.judge,
        max_pairs=args.max_pairs,
    )
    if args.dry_run:
        print(json.dumps(report, sort_keys=True))
        return 0

    if args.judge == PRIMARY_KEY:
        assert config.primary.local_path is not None
        runtime_model_path = args.runtime_model_path or config.primary.local_path
        resolved = validate_model_assets(runtime_model_path)
        handle = load_local_qwen(resolved)
        try:
            if handle.model is None or handle.tokenizer is None:
                raise ValueError("Octopus judge model failed to load")
            judge = OctopusLocalJudge(
                model=handle.model,
                tokenizer=handle.tokenizer,
                revision=resolved.revision,
            )
            report = run_with_judge(
                config,
                pairs_path,
                output_path,
                judge=judge,
                max_pairs=args.max_pairs,
            )
        finally:
            handle.close()
        report["runtime_model_path"] = str(runtime_model_path)
    else:
        assert config.secondary.endpoint is not None and config.secondary.model is not None
        runtime_endpoint = args.runtime_endpoint or config.secondary.endpoint
        with Qwen32CompatJudge(
            endpoint=runtime_endpoint,
            model=config.secondary.model,
        ) as judge:
            report = run_with_judge(
                config,
                pairs_path,
                output_path,
                judge=judge,
                max_pairs=args.max_pairs,
            )
        report["runtime_endpoint"] = runtime_endpoint
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
