"""Materialize and generate accepted FOL local perturbations without printing text."""

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
from benchmark.safety_eval.fol_boundary import random_joint_direction
from benchmark.safety_eval.fol_runtime import (
    PerturbationScheduleRow,
    select_id_shard,
    select_accepted_perturbations,
)
from benchmark.safety_eval.io import JsonlLedger, read_jsonl
from benchmark.safety_eval.materialization import materialize_checkpoint
from benchmark.safety_eval.pipeline import (
    generate_materialized_records,
    write_materialization_records,
)
from benchmark.safety_eval.runtime import validate_model_assets
from benchmark.safety_eval.schema import BenchmarkExample, OptimizationRecord, RecordStatus, stable_id


ROOT = Path(__file__).resolve().parents[1]


def _final_records(root: Path, source: str) -> tuple[dict[str, BenchmarkExample], dict[str, OptimizationRecord]]:
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


def _selection_bands(root: Path, sources: tuple[str, ...]) -> dict[tuple[str, str], str]:
    payload = json.loads((root / "manifests" / "validation_selection.json").read_text(encoding="utf-8"))
    source_map = payload.get("sources")
    if not isinstance(source_map, dict):
        raise ValueError("FOL validation selection has no source map")
    bands: dict[tuple[str, str], str] = {}
    for source in sources:
        values = source_map.get(source)
        if not isinstance(values, dict):
            raise ValueError(f"FOL validation selection is missing {source}")
        for band in ("low", "middle", "high"):
            ids = values.get(band)
            if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
                raise ValueError(f"FOL validation selection has invalid {source}/{band} IDs")
            for sample_id in ids:
                key = (source, sample_id)
                if key in bands:
                    raise ValueError("FOL validation sample appears in multiple bands")
                bands[key] = band
    return bands


def _accepted_metadata(root: Path, *, sources: tuple[str, ...], directions_per_radius: int) -> tuple[dict[str, object], ...]:
    schedule_rows = [row for row in read_jsonl(root / "perturbation_schedule.jsonl") if row.get("source") in sources]
    outcomes = {
        str(row.get("perturbation_id")): row
        for row in read_jsonl(root / "perturbation_semantic_outcomes.jsonl")
        if row.get("source") in sources and isinstance(row.get("perturbation_id"), str)
    }
    if len(outcomes) != len(schedule_rows):
        raise ValueError("FOL semantic outcomes are incomplete")
    bands = _selection_bands(root, sources)
    by_source: dict[str, list[PerturbationScheduleRow]] = {source: [] for source in sources}
    accepted_by_source: dict[str, dict[str, bool]] = {source: {} for source in sources}
    source_by_id: dict[str, str] = {}
    for row in schedule_rows:
        try:
            source = row["source"]
            schedule = PerturbationScheduleRow(
                perturbation_id=str(row["perturbation_id"]),
                sample_id=str(row["sample_id"]),
                radius=float(row["radius"]),
                direction_index=int(row["direction_index"]),
                direction_seed=int(row["direction_seed"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("FOL perturbation schedule is invalid") from error
        if not isinstance(source, str) or source not in by_source:
            raise ValueError("FOL perturbation schedule source is invalid")
        outcome = outcomes.get(schedule.perturbation_id)
        if outcome is None or type(outcome.get("accepted")) is not bool:
            raise ValueError("FOL semantic outcome is invalid")
        by_source[source].append(schedule)
        accepted_by_source[source][schedule.perturbation_id] = bool(outcome["accepted"])
        source_by_id[schedule.perturbation_id] = source

    metadata: list[dict[str, object]] = []
    for source in sources:
        selected = select_accepted_perturbations(
            by_source[source],
            accepted_by_id=accepted_by_source[source],
            directions_per_radius=directions_per_radius,
        )
        for schedule in selected:
            outcome = outcomes[schedule.perturbation_id]
            similarity = outcome.get("semantic_similarity")
            if isinstance(similarity, bool) or not isinstance(similarity, (int, float)) or not math.isfinite(float(similarity)):
                raise ValueError("accepted FOL semantic outcome has invalid similarity")
            band = bands.get((source, schedule.sample_id))
            if band is None:
                raise ValueError("accepted FOL direction is outside the validation selection")
            metadata.append({
                "perturbation_id": schedule.perturbation_id,
                "source": source,
                "sample_id": schedule.sample_id,
                "band": band,
                "kind": "perturbation",
                "radius": schedule.radius,
                "direction_index": schedule.direction_index,
                "direction_seed": schedule.direction_seed,
                "semantic_similarity": float(similarity),
            })
    for (source, sample_id), band in sorted(bands.items()):
        metadata.append({
            "perturbation_id": stable_id("fol-baseline", {"source": source, "sample_id": sample_id}),
            "source": source,
            "sample_id": sample_id,
            "band": band,
            "kind": "baseline",
            "radius": None,
            "direction_index": None,
            "direction_seed": None,
            "semantic_similarity": 1.0,
        })
    return tuple(sorted(metadata, key=lambda row: str(row["perturbation_id"])))


def _perturbed_payload(
    optimization: OptimizationRecord,
    metadata: dict[str, object],
    *,
    embeddings: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not optimization.state_path:
        raise ValueError("FOL terminal state path is missing")
    payload = torch.load(optimization.state_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("z"), torch.Tensor) or not isinstance(payload.get("u"), torch.Tensor):
        raise ValueError("FOL terminal state has invalid editable tensors")
    z = payload["z"].to(device=embeddings.device, dtype=embeddings.dtype)
    u = payload["u"].to(device=embeddings.device, dtype=embeddings.dtype)
    if metadata["kind"] == "perturbation":
        radius = metadata.get("radius")
        seed = metadata.get("direction_seed")
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("FOL perturbation metadata is invalid")
        direction = torch.Generator(device="cpu").manual_seed(seed)
        dz, du = random_joint_direction(tuple(z.shape), tuple(u.shape), direction)
        z = z + float(radius) * dz.to(device=z.device, dtype=z.dtype)
        u = u + float(radius) * du.to(device=u.device, dtype=u.dtype)
    return {"z": z, "u": u}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    sources = tuple(args.source or config.fol.sources)
    if set(sources) - set(config.fol.sources):
        raise ValueError("FOL generation requested an unconfigured source")
    root = ROOT / config.run.output_root / "fol_boundary"
    metadata = _accepted_metadata(root, sources=sources, directions_per_radius=config.fol.directions_per_radius)
    selected_ids = set(select_id_shard(
        tuple(str(row["perturbation_id"]) for row in metadata),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    ))
    metadata = tuple(row for row in metadata if str(row["perturbation_id"]) in selected_ids)
    metadata_ledger = JsonlLedger(root / "selected_perturbations.jsonl", key_fields=("perturbation_id",))
    for row in metadata:
        metadata_ledger.append_once(row)
    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    run_id, config_hash = run_manifest.get("run_id"), run_manifest.get("config_hash")
    if not isinstance(run_id, str) or not isinstance(config_hash, str):
        raise ValueError("FOL run manifest is invalid")
    model_path = config.models.surrogate.local_path
    if model_path is None:
        raise ValueError("FOL generation requires a local target")
    resolved = validate_model_assets(model_path)
    handle = load_local_qwen(resolved)
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("local FOL target did not load")
        embeddings = handle.model.get_input_embeddings().weight.detach()
        source_records = {source: _final_records(root, source) for source in sources}
        materializations = []
        for row in metadata:
            source, sample_id, perturbation_id = row["source"], row["sample_id"], row["perturbation_id"]
            if not all(isinstance(value, str) for value in (source, sample_id, perturbation_id)):
                raise ValueError("FOL metadata identity is invalid")
            example, optimization = source_records[source][0][sample_id], source_records[source][1][sample_id]
            materialization = materialize_checkpoint(
                state_payload=_perturbed_payload(optimization, row, embeddings=embeddings),
                vocabulary_embeddings=embeddings,
                tokenizer=handle.tokenizer,
                schema_version=config.run.schema_version,
                run_id=run_id,
                config_hash=config_hash,
                sample_id=perturbation_id,
                source=source,
                method="fol_boundary",
                checkpoint=0,
                original_prompt=example.attack_text,
                category=example.risk_category,
                semantic_similarity=float(row["semantic_similarity"]),
                semantic_threshold=0.0,
            )
            materializations.append(materialization.model_dump(mode="json"))
        materialized = write_materialization_records(root, materializations)
        generated = generate_materialized_records(
            root,
            materializations,
            model=handle.model,
            tokenizer=handle.tokenizer,
            target_key=config.models.targets[0].key,
            target_revision=resolved.revision,
            max_new_tokens=config.judging.max_new_tokens,
        )
    finally:
        handle.close()
    print(json.dumps({
        "generated": generated.written_records,
        "materialized": materialized.written_records,
        "selected": len(metadata),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
