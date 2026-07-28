"""Freeze FOL radius-calibration direction identities before model evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.fol_runtime import build_perturbation_schedule
from benchmark.safety_eval.io import atomic_write_json, atomic_write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = ROOT / config.run.output_root / "fol_boundary"
    selection_path = root / "manifests" / "validation_selection.json"
    schedule_path = root / "radius_calibration_schedule.jsonl"
    metadata_path = root / "radius_calibration_schedule.json"
    if schedule_path.is_file() and metadata_path.is_file():
        print(json.dumps({"artifact": str(schedule_path), "status": "existing"}, sort_keys=True))
        return 0
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    sources = selection.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("FOL validation selection has no source map")
    radii = tuple(config.fol.base_radius_candidates)
    rows = []
    for source in config.fol.sources:
        source_selection = sources.get(source)
        if not isinstance(source_selection, dict):
            raise ValueError(f"FOL selection is missing {source}")
        ids = source_selection.get("radius_calibration")
        if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
            raise ValueError(f"FOL selection has invalid {source}/radius_calibration IDs")
        if len(ids) != config.fol.radius_calibration_per_source:
            raise ValueError(f"FOL selection has wrong radius-calibration count for {source}")
        for row in build_perturbation_schedule(
            sample_ids=tuple(ids),
            radii=radii,
            directions_per_radius=config.fol.directions_per_radius,
            seed=config.run.seed,
        ):
            rows.append({"source": source, **row.__dict__})
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(schedule_path, rows)
    atomic_write_json(metadata_path, {
        "config_hash": selection.get("config_hash"),
        "direction_count": len(rows),
        "radii": list(radii),
        "sources": list(config.fol.sources),
    })
    print(json.dumps({"artifact": str(schedule_path), "direction_count": len(rows), "status": "frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
