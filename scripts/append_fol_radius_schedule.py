"""Append deterministic, content-free FOL perturbation rows for new radii."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark.safety_eval.fol_runtime import build_perturbation_schedule
from benchmark.safety_eval.io import JsonlLedger, read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fol-root", type=Path, required=True)
    parser.add_argument("--radii", type=float, nargs="+", required=True)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    if args.directions < 1:
        raise ValueError("directions must be positive")
    root = args.fol_root
    selection = json.loads((root / "manifests" / "validation_selection.json").read_text(encoding="utf-8"))
    source_map = selection.get("sources")
    if not isinstance(source_map, dict):
        raise ValueError("validation selection has no source map")
    ledger = JsonlLedger(root / "perturbation_schedule.jsonl", key_fields=("perturbation_id",))
    appended = 0
    for source, values in sorted(source_map.items()):
        if not isinstance(source, str) or not isinstance(values, dict):
            raise ValueError("validation selection has invalid source data")
        sample_ids = []
        band_by_id: dict[str, str] = {}
        for band in ("low", "middle", "high"):
            ids = values.get(band)
            if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
                raise ValueError("validation selection has invalid bands")
            for sample_id in ids:
                if sample_id in band_by_id:
                    raise ValueError("validation sample appears in multiple bands")
                sample_ids.append(sample_id)
                band_by_id[sample_id] = band
        source_seed = int(hashlib.sha256(f"{args.seed}|{source}".encode()).hexdigest()[:16], 16) % (2**31)
        for row in build_perturbation_schedule(
            sample_ids=sample_ids,
            radii=args.radii,
            directions_per_radius=args.directions,
            seed=source_seed,
        ):
            payload = {"source": source, "band": band_by_id[row.sample_id], **row.__dict__}
            appended += int(ledger.append_once(payload))
    print(json.dumps({"appended": appended, "root": str(root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
