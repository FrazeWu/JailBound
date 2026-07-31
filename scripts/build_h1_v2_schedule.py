"""Freeze H1-v2 oversampled direction identities before semantic evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.safety_eval.config import load_h1_v2_config
from benchmark.safety_eval.fol_runtime import build_h1_v2_direction_schedule
from benchmark.safety_eval.io import atomic_write_json, atomic_write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_h1_v2_config(args.config)
    root = ROOT / config.h1_v2.output_root
    selection_path = root / "manifests" / "h1_v2_validation_selection.json"
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("H1-v2 validation selection has no source map")
    rows = []
    for source in config.h1_v2.sources:
        selected = sources.get(source)
        if not isinstance(selected, dict):
            raise ValueError(f"H1-v2 validation selection is missing {source}")
        ids = [
            sample_id
            for band in ("low", "middle", "high")
            for sample_id in selected.get(band, [])
            if isinstance(sample_id, str)
        ]
        if len(ids) != 37 or len(set(ids)) != 37:
            raise ValueError(f"H1-v2 validation selection has invalid IDs for {source}")
        rows.extend({"source": source, **row.__dict__} for row in build_h1_v2_direction_schedule(
            sample_ids=ids,
            radii=config.h1_v2.radius_candidates,
            max_direction_attempts=config.h1_v2.max_direction_attempts,
            seed=config.base.run.seed,
        ))
    atomic_write_jsonl(root / "direction_schedule.jsonl", rows)
    atomic_write_json(root / "direction_schedule.json", {
        "config_hash": payload.get("config_hash"),
        "accepted_directions": config.h1_v2.accepted_directions,
        "max_direction_attempts": config.h1_v2.max_direction_attempts,
        "radii": list(config.h1_v2.radius_candidates),
        "sources": list(config.h1_v2.sources),
    })
    print(json.dumps({"direction_count": len(rows), "status": "frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
