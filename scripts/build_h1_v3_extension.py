"""Freeze H1-v3 source provenance and oversampled new-radius directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.safety_eval.config import load_h1_v3_config
from benchmark.safety_eval.h1_v3_runtime import (
    H1V3Paths,
    build_h1_v3_contract,
    build_h1_v3_schedule,
    frozen_h1_v2_selected_ids,
)
from benchmark.safety_eval.io import atomic_write_json, atomic_write_jsonl, canonical_hash


ROOT = Path(__file__).resolve().parents[1]


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"H1-v3 {label} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"H1-v3 {label} must be an object")
    return value


def _write_or_validate(path: Path, payload: dict[str, object], *, label: str) -> None:
    if path.exists():
        if _read_object(path, label=label) != payload:
            raise ValueError(f"H1-v3 existing {label} differs from frozen inputs")
        return
    atomic_write_json(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_h1_v3_config(args.config)
    paths = H1V3Paths(
        source_root=ROOT / config.h1_v3.source_root,
        output_root=ROOT / config.h1_v3.output_root,
    )
    source_run = _read_object(paths.source_run_manifest(), label="source run manifest")
    source_hash = source_run.get("config_hash")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("H1-v3 source run manifest has no configuration hash")
    selected_ids = frozen_h1_v2_selected_ids(paths.source_root)
    if len(selected_ids) != 74:
        raise ValueError("H1-v3 requires exactly 37 frozen prompts per source")

    contract = build_h1_v3_contract(
        source_hash=source_hash,
        selected_ids=selected_ids,
        radii=config.h1_v3.radius_candidates,
    )
    source_contract = {
        **contract,
        "source_root": str(config.h1_v3.source_root),
        "endpoint_primary": "mean_unsafe_score_over_new_radii",
        "endpoint_secondary": "unsafe_transition_rate_over_new_radii",
        "h2_extension_radii": [0.025, 0.05, 0.1, 0.2, 0.4, 0.6],
    }
    _write_or_validate(paths.output_root / "source_contract.json", source_contract, label="source contract")
    output_config_hash = canonical_hash(config.h1_v3.model_dump(mode="json"))
    run_manifest = {
        "run_id": "fol_h1_v3_local_radius_extension",
        "config_hash": output_config_hash,
        "analysis_status": "follow_up_extension",
        "source_h1_v2_config_hash": source_hash,
        "selected_ids_hash": contract["selected_ids_hash"],
        "new_radii": list(config.h1_v3.radius_candidates),
    }
    _write_or_validate(paths.output_root / "run_manifest.json", run_manifest, label="run manifest")

    rows = []
    for source in config.h1_v3.sources:
        sample_ids = tuple(sample_id for selected_source, sample_id in selected_ids if selected_source == source)
        rows.extend(
            {"source": source, **row.__dict__}
            for row in build_h1_v3_schedule(
                sample_ids=sample_ids,
                radii=config.h1_v3.radius_candidates,
                attempts=config.h1_v3.max_direction_attempts,
                seed=config.base.run.seed,
            )
        )
    schedule_metadata = {
        "source_h1_v2_config_hash": source_hash,
        "selected_ids_hash": contract["selected_ids_hash"],
        "accepted_directions": config.h1_v3.accepted_directions,
        "max_direction_attempts": config.h1_v3.max_direction_attempts,
        "radii": list(config.h1_v3.radius_candidates),
        "sources": list(config.h1_v3.sources),
    }
    if paths.schedule().exists():
        existing = paths.schedule().read_text(encoding="utf-8")
        expected = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
        if existing != expected:
            raise ValueError("H1-v3 existing direction schedule differs from frozen inputs")
    else:
        atomic_write_jsonl(paths.schedule(), rows)
    _write_or_validate(paths.schedule_metadata(), schedule_metadata, label="schedule metadata")
    print(json.dumps({"direction_count": len(rows), "status": "frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
