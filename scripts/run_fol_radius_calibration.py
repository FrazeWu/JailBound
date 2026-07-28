"""Calibrate the FOL local radius with semantic acceptance only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.fol_records import resolved_terminal_payloads
from benchmark.safety_eval.fol_boundary import random_joint_direction, select_base_radius
from benchmark.safety_eval.fol_runtime import PerturbationScheduleRow, select_schedule_shard
from benchmark.safety_eval.io import JsonlLedger, atomic_write_json, read_jsonl
from benchmark.safety_eval.materialization import materialize_continuous_state
from benchmark.safety_eval.objective import EditableState
from benchmark.safety_eval.runtime import validate_model_assets
from benchmark.safety_eval.schema import BenchmarkExample, OptimizationRecord, RecordStatus
from benchmark.safety_eval.semantic import QwenHiddenMeanEncoder


ROOT = Path(__file__).resolve().parents[1]


def _artifact_name(value: str, *, field: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{field} must be a single file name")
    return value


def _threshold(root: Path) -> float:
    payload = json.loads((root.parent / "manifests" / "semantic_calibration.json").read_text(encoding="utf-8"))
    value = payload.get("threshold")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
        raise ValueError("semantic calibration threshold is invalid")
    return float(value)


def _records(root: Path, source: str) -> tuple[dict[str, BenchmarkExample], dict[str, OptimizationRecord]]:
    examples = {
        row.example_id: row
        for row in (
            BenchmarkExample.model_validate(row)
            for row in read_jsonl(root / "manifests" / f"controlled_{source}.jsonl")
        )
    }
    optimized = {
        row.sample_id: row
        for row in (
            OptimizationRecord.model_validate(row)
            for row in resolved_terminal_payloads(root, source).values()
        )
        if row.checkpoint == 100 and row.status is RecordStatus.complete and row.state_path
    }
    if set(examples) != set(optimized):
        raise ValueError(f"FOL terminal O+ states are incomplete for {source}")
    return examples, optimized


def _candidate_similarity(
    *,
    example: BenchmarkExample,
    state_path: str,
    radius: float,
    direction_seed: int,
    tokenizer: Any,
    embeddings: torch.Tensor,
    encoder: QwenHiddenMeanEncoder,
) -> float:
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    z, u = payload.get("z"), payload.get("u")
    if not isinstance(z, torch.Tensor) or not isinstance(u, torch.Tensor):
        raise ValueError("FOL checkpoint has no editable tensors")
    generator = torch.Generator(device="cpu").manual_seed(direction_seed)
    dz, du = random_joint_direction(tuple(z.shape), tuple(u.shape), generator)
    device = embeddings.device
    base_z = z.to(device=device, dtype=embeddings.dtype)
    base_u = u.to(device=device, dtype=embeddings.dtype)
    state = EditableState(
        z=base_z + radius * dz.to(device=device, dtype=embeddings.dtype),
        u=base_u + radius * du.to(device=device, dtype=embeddings.dtype),
        z0=base_z,
        u0=base_u,
    )
    projected = materialize_continuous_state(
        state,
        embeddings,
        forbidden_token_ids=tuple(int(value) for value in tokenizer.all_special_ids),
    )
    prefix = str(tokenizer.decode(list(projected.prefix_token_ids), skip_special_tokens=True)).strip()
    suffix = str(tokenizer.decode(list(projected.seed_token_ids), skip_special_tokens=True)).strip()
    reconstructed = " ".join(part for part in (prefix, example.attack_text.strip(), suffix) if part)
    vectors = encoder.encode([example.attack_text, reconstructed])
    return float(vectors[0] @ vectors[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--schedule-name", default="radius_calibration_schedule.jsonl")
    parser.add_argument("--outcomes-name", default="radius_calibration_outcomes.jsonl")
    parser.add_argument("--source", action="append")
    parser.add_argument("--skip-base-radius", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    root = ROOT / config.run.output_root / "fol_boundary"
    threshold = _threshold(root)
    schedule_name = _artifact_name(args.schedule_name, field="schedule name")
    outcomes_name = _artifact_name(args.outcomes_name, field="outcomes name")
    sources = tuple(args.source or config.fol.sources)
    if set(sources) - set(config.fol.sources):
        raise ValueError("FOL semantic evaluation requested an unconfigured source")
    schedule = [
        row for row in read_jsonl(root / schedule_name)
        if row.get("source") in sources
    ]
    if not schedule:
        raise ValueError("FOL semantic-evaluation schedule is missing for the selected sources")
    schedule_identities = tuple(
        PerturbationScheduleRow(
            perturbation_id=str(row["perturbation_id"]),
            sample_id=str(row["sample_id"]),
            radius=float(row["radius"]),
            direction_index=int(row["direction_index"]),
            direction_seed=int(row["direction_seed"]),
        )
        for row in schedule
    )
    selected_ids = {
        row.perturbation_id
        for row in select_schedule_shard(
            schedule_identities,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    }
    schedule = [row for row in schedule if row.get("perturbation_id") in selected_ids]
    model_path = config.models.surrogate.local_path
    if model_path is None:
        raise ValueError("FOL radius calibration requires a local surrogate")
    resolved = validate_model_assets(model_path)
    ledger = JsonlLedger(root / outcomes_name, key_fields=("perturbation_id",))
    handle = load_local_qwen(resolved)
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("local FOL surrogate did not load")
        embeddings = handle.model.get_input_embeddings().weight.detach()
        encoder = QwenHiddenMeanEncoder(model_path, tokenizer=handle.tokenizer, model=handle.model, revision=resolved.revision)
        source_records = {source: _records(root, source) for source in sources}
        for row in schedule:
            perturbation_id = row.get("perturbation_id")
            if not isinstance(perturbation_id, str):
                raise ValueError("FOL schedule has invalid perturbation ID")
            if ledger.contains_key({"perturbation_id": perturbation_id}):
                continue
            source = row.get("source")
            sample_id = row.get("sample_id")
            try:
                if not isinstance(source, str) or not isinstance(sample_id, str):
                    raise ValueError("FOL schedule has invalid source or sample ID")
                example, optimized = source_records[source][0][sample_id], source_records[source][1][sample_id]
                radius = float(row["radius"])
                similarity = _candidate_similarity(
                    example=example,
                    state_path=str(optimized.state_path),
                    radius=radius,
                    direction_seed=int(row["direction_seed"]),
                    tokenizer=handle.tokenizer,
                    embeddings=embeddings,
                    encoder=encoder,
                )
                result = dict(row, semantic_similarity=similarity, accepted=similarity >= threshold, status="complete", failure_reason=None)
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                result = dict(row, semantic_similarity=None, accepted=False, status="failed", failure_reason=type(error).__name__)
            ledger.append_once(result)
    finally:
        handle.close()
    outcomes = read_jsonl(root / outcomes_name)
    if args.skip_base_radius:
        print(json.dumps({"outcome_count": len(outcomes), "status": "complete"}, sort_keys=True))
        return 0
    acceptance: dict[float, list[bool]] = {}
    source_acceptance: dict[str, dict[float, list[bool]]] = {}
    for row in outcomes:
        radius = row.get("radius")
        accepted = row.get("accepted")
        source = row.get("source")
        if isinstance(radius, (int, float)) and type(accepted) is bool and isinstance(source, str):
            acceptance.setdefault(float(radius), []).append(accepted)
            source_acceptance.setdefault(source, {}).setdefault(float(radius), []).append(accepted)
    base_radius = select_base_radius(
        acceptance,
        minimum_rate=0.8,
        source_acceptance=source_acceptance,
        minimum_source_rate=0.75,
    )
    atomic_write_json(root / "base_radius.json", {
        "base_radius": base_radius,
        "semantic_threshold": threshold,
        "acceptance": {
            str(radius): {"accepted": sum(values), "total": len(values)}
            for radius, values in sorted(acceptance.items())
        },
        "source_acceptance": {
            source: {
                str(radius): {"accepted": sum(values), "total": len(values)}
                for radius, values in sorted(by_radius.items())
            }
            for source, by_radius in sorted(source_acceptance.items())
        },
    })
    print(json.dumps({"base_radius": base_radius, "outcome_count": len(outcomes)}, sort_keys=True))
    return 0 if base_radius is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
