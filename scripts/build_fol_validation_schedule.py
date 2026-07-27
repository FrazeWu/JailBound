"""Freeze FOL validation directions after the semantic radius is calibrated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.reviewer_eval.config import load_config
from benchmark.reviewer_eval.fol_runtime import build_perturbation_schedule
from benchmark.reviewer_eval.io import atomic_write_json, atomic_write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = ROOT / config.run.output_root / "fol_boundary"
    base_path = root / "base_radius.json"
    selection_path = root / "manifests" / "validation_selection.json"
    schedule_path = root / "perturbation_schedule.jsonl"
    metadata_path = root / "perturbation_schedule.json"
    if schedule_path.is_file() and metadata_path.is_file():
        print(json.dumps({"artifact": str(schedule_path), "status": "existing"}, sort_keys=True))
        return 0
    base_payload = json.loads(base_path.read_text(encoding="utf-8"))
    base_radius = base_payload.get("base_radius")
    if not isinstance(base_radius, (int, float)) or isinstance(base_radius, bool) or base_radius <= 0:
        raise ValueError("FOL base-radius artifact is invalid or inconclusive")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    sources = selection.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("FOL validation selection has no source map")
    radii = tuple(float(base_radius) * multiplier for multiplier in config.fol.radius_multipliers)
    rows = []
    for source in config.fol.sources:
        source_selection = sources.get(source)
        if not isinstance(source_selection, dict):
            raise ValueError(f"FOL selection is missing {source}")
        sample_ids = []
        for band in ("low", "middle", "high"):
            values = source_selection.get(band)
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError(f"FOL selection has invalid {source}/{band} IDs")
            sample_ids.extend(values)
        if len(sample_ids) != config.fol.validation_per_source:
            raise ValueError(f"FOL selection has wrong validation count for {source}")
        rows.extend(
            {"source": source, **row.__dict__}
            for row in build_perturbation_schedule(
                sample_ids=tuple(sample_ids),
                radii=radii,
                directions_per_radius=config.fol.max_direction_attempts,
                seed=config.run.seed,
            )
        )
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(schedule_path, rows)
    atomic_write_json(metadata_path, {
        "config_hash": selection.get("config_hash"),
        "base_radius": float(base_radius),
        "direction_count": len(rows),
        "accepted_directions_per_radius": config.fol.directions_per_radius,
        "max_attempts_per_radius": config.fol.max_direction_attempts,
        "radii": list(radii),
    })
    print(json.dumps({"artifact": str(schedule_path), "direction_count": len(rows), "status": "frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
