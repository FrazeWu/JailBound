"""Write the content-free, one-candidate H1-v2 OOM exclusion amendment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.safety_eval.config import load_h1_v2_config
from benchmark.safety_eval.fol_records import resolved_terminal_payloads
from benchmark.safety_eval.fol_runtime import select_h1_v2_eligible_ids, select_h1_v2_oom_recovery_ids
from benchmark.safety_eval.io import atomic_write_json, read_jsonl
from benchmark.safety_eval.schema import BenchmarkExample


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_h1_v2_config(args.config)
    root = ROOT / config.h1_v2.output_root
    run = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    config_hash = run.get("config_hash")
    if not isinstance(config_hash, str):
        raise ValueError("H1-v2 run manifest has no configuration hash")
    sources: dict[str, list[str]] = {}
    for source in config.h1_v2.sources:
        manifest_ids = tuple(
            row.example_id
            for row in (
                BenchmarkExample.model_validate(item)
                for item in read_jsonl(root / "manifests" / f"controlled_{source}.jsonl")
            )
        )
        terminal_ids = tuple(resolved_terminal_payloads(root, source))
        unresolved = tuple(sorted(set(manifest_ids) - set(terminal_ids)))
        primary = read_jsonl(root / "optimization" / source / "jailbound_o_plus" / "records.jsonl")
        recoverable = set(select_h1_v2_oom_recovery_ids(primary))
        if not set(unresolved) <= recoverable:
            raise ValueError("H1-v2 unresolved candidate is not an all-checkpoint primary OOM")
        select_h1_v2_eligible_ids(
            manifest_ids=manifest_ids, terminal_ids=terminal_ids, computational_exclusions=unresolved,
        )
        sources[source] = list(unresolved)
    payload = {
        "config_hash": config_hash,
        "criterion": "all_primary_checkpoints_oom_without_successful_registered_recovery",
        "original_candidate_count_per_source": config.h1_v2.candidate_count,
        "sources": sources,
    }
    destination = root / "manifests" / "h1_v2_computational_exclusions.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("H1-v2 computational exclusion artifact differs from the verified amendment")
    else:
        atomic_write_json(destination, payload)
    print(json.dumps({"sources": {source: len(ids) for source, ids in sources.items()}, "status": "frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
