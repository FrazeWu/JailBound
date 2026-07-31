"""Validate H1-v3 semantic acceptance and local state change before generation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from benchmark.safety_eval.config import load_h1_v3_config
from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.fol_boundary import random_joint_direction
from benchmark.safety_eval.fol_runtime import PerturbationScheduleRow, select_first_accepted_directions, select_schedule_shard
from benchmark.safety_eval.h1_v3_runtime import H1V3Paths, frozen_h1_v2_records
from benchmark.safety_eval.io import JsonlLedger, atomic_write_json, atomic_write_jsonl, read_jsonl
from benchmark.safety_eval.runtime import validate_model_assets
from benchmark.safety_eval.semantic import QwenHiddenMeanEncoder

from run_fol_radius_calibration import _candidate_similarity


ROOT = Path(__file__).resolve().parents[1]


def _semantic_threshold(config: object) -> float:
    path = ROOT / config.base.semantic.threshold_artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    threshold = payload.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("H1-v3 semantic threshold is invalid")
    return float(threshold)


def _relative_state_change(state_path: str, *, radius: float, direction_seed: int) -> float:
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    z, u = payload.get("z"), payload.get("u")
    if not isinstance(z, torch.Tensor) or not isinstance(u, torch.Tensor):
        raise ValueError("H1-v3 checkpoint has no editable tensors")
    generator = torch.Generator(device="cpu").manual_seed(direction_seed)
    dz, du = random_joint_direction(tuple(z.shape), tuple(u.shape), generator)
    numerator = torch.sqrt(torch.sum((radius * dz.float()) ** 2) + torch.sum((radius * du.float()) ** 2))
    denominator = torch.sqrt(torch.sum(z.float() ** 2) + torch.sum(u.float() ** 2))
    value = float((numerator / denominator.clamp_min(1e-12)).cpu())
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("H1-v3 relative state change is invalid")
    return value


def _finalize(paths: H1V3Paths, config: object, schedule: list[dict[str, object]]) -> None:
    outcomes = {str(row.get("perturbation_id")): row for row in read_jsonl(paths.semantic_ledger())}
    schedule_ids = {str(row.get("perturbation_id")) for row in schedule}
    if set(outcomes) != schedule_ids:
        raise ValueError("H1-v3 semantic outcomes are incomplete")
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    selected_rows: list[dict[str, object]] = []
    insufficiencies: list[dict[str, object]] = []
    for radius in config.h1_v3.radius_candidates:
        for source in config.h1_v3.sources:
            rows = [row for row in schedule if row.get("source") == source and float(row["radius"]) == radius]
            accepted = [outcomes[str(row["perturbation_id"])] for row in rows]
            if not rows or any(type(row.get("accepted")) is not bool or row.get("status") != "complete" for row in accepted):
                raise ValueError("H1-v3 semantic outcomes are invalid")
            acceptance_rate = sum(bool(row["accepted"]) for row in accepted) / len(accepted)
            maximum_change = max(float(row["relative_state_change"]) for row in accepted)
            summary.setdefault(source, {})[f"{radius:.17g}"] = {
                "total": len(accepted),
                "accepted": sum(bool(row["accepted"]) for row in accepted),
                "acceptance_rate": acceptance_rate,
                "max_relative_state_change": maximum_change,
            }
            if acceptance_rate < config.h1_v3.semantic_acceptance_floor or maximum_change > config.h1_v3.relative_state_change_cap:
                raise ValueError(f"H1-v3 radius {radius} failed the local semantic gate for {source}")
    for source in config.h1_v3.sources:
        rows = [
            PerturbationScheduleRow(
                perturbation_id=str(row["perturbation_id"]), sample_id=str(row["sample_id"]), radius=float(row["radius"]),
                direction_index=int(row["direction_index"]), direction_seed=int(row["direction_seed"]),
            )
            for row in schedule if row.get("source") == source
        ]
        accepted_by_id = {row.perturbation_id: bool(outcomes[row.perturbation_id]["accepted"]) for row in rows}
        selection = select_first_accepted_directions(rows, accepted_by_id=accepted_by_id, required_count=config.h1_v3.accepted_directions)
        insufficiencies.extend(
            {"source": source, "sample_id": sample_id, "radius": radius, "accepted": count}
            for sample_id, radius, count in selection.insufficient
        )
        selected_rows.extend({"source": source, **row.__dict__} for row in selection.accepted)
    atomic_write_json(paths.output_root / "locality_calibration.json", {
        "analysis_status": "follow_up_extension",
        "source_h1_v2_config_hash": json.loads(paths.source_run_manifest().read_text(encoding="utf-8"))["config_hash"],
        "semantic_threshold": _semantic_threshold(config),
        "semantic_acceptance_floor": config.h1_v3.semantic_acceptance_floor,
        "relative_state_change_cap": config.h1_v3.relative_state_change_cap,
        "r_local": list(config.h1_v3.radius_candidates),
        "source_radius_summary": summary,
        "behavior_labels_read": False,
    })
    atomic_write_jsonl(paths.output_root / "direction_insufficiencies.jsonl", insufficiencies)
    if insufficiencies:
        raise ValueError("H1-v3 has prompt-radius groups with fewer than 32 accepted directions")
    atomic_write_jsonl(paths.selected_directions(), selected_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("H1-v3 locality shard settings are invalid")
    config = load_h1_v3_config(args.config)
    paths = H1V3Paths(ROOT / config.h1_v3.source_root, ROOT / config.h1_v3.output_root)
    schedule = read_jsonl(paths.schedule())
    sources = tuple(args.source or config.h1_v3.sources)
    if set(sources) - set(config.h1_v3.sources):
        raise ValueError("H1-v3 locality calibration requested an unconfigured source")
    if args.finalize:
        _finalize(paths, config, schedule)
        print(json.dumps({"status": "frozen"}, sort_keys=True))
        return 0
    selected_schedule = [
        PerturbationScheduleRow(
            perturbation_id=str(row["perturbation_id"]), sample_id=str(row["sample_id"]), radius=float(row["radius"]),
            direction_index=int(row["direction_index"]), direction_seed=int(row["direction_seed"]),
        )
        for row in schedule if row.get("source") in sources
    ]
    selected_ids = {
        row.perturbation_id
        for row in select_schedule_shard(selected_schedule, shard_index=args.shard_index, shard_count=args.shard_count)
    }
    rows = [row for row in schedule if row.get("perturbation_id") in selected_ids]
    model_path = config.base.models.surrogate.local_path
    if model_path is None:
        raise ValueError("H1-v3 locality calibration requires a local surrogate")
    threshold = _semantic_threshold(config)
    handle = load_local_qwen(validate_model_assets(model_path), attention_backend=config.base.run.attention_implementation)
    ledger = JsonlLedger(paths.semantic_ledger(), key_fields=("perturbation_id",))
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("H1-v3 local surrogate did not load")
        embeddings = handle.model.get_input_embeddings().weight.detach()
        encoder = QwenHiddenMeanEncoder(model_path, tokenizer=handle.tokenizer, model=handle.model)
        source_records = {source: frozen_h1_v2_records(paths.source_root, source) for source in sources}
        for row in rows:
            perturbation_id = str(row["perturbation_id"])
            if ledger.contains_key({"perturbation_id": perturbation_id}):
                continue
            try:
                source, sample_id = str(row["source"]), str(row["sample_id"])
                example, optimized = source_records[source][0][sample_id], source_records[source][1][sample_id]
                similarity = _candidate_similarity(
                    example=example, state_path=str(optimized.state_path), radius=float(row["radius"]),
                    direction_seed=int(row["direction_seed"]), tokenizer=handle.tokenizer,
                    embeddings=embeddings, encoder=encoder,
                )
                result = dict(
                    row, semantic_similarity=similarity, accepted=similarity >= threshold,
                    relative_state_change=_relative_state_change(str(optimized.state_path), radius=float(row["radius"]), direction_seed=int(row["direction_seed"])),
                    status="complete", failure_reason=None,
                )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                result = dict(
                    row, semantic_similarity=None, accepted=False, relative_state_change=None,
                    status="failed", failure_reason=type(error).__name__,
                )
            ledger.append_once(result)
    finally:
        handle.close()
    print(json.dumps({"selected": len(rows), "status": "evaluated"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
