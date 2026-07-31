"""Resume FOL interpolation response generation from persisted materializations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.io import read_jsonl
from benchmark.safety_eval.pipeline import generate_materialized_records
from benchmark.safety_eval.runtime import validate_model_assets


ROOT = Path(__file__).resolve().parents[1]


def interpolation_materializations(root: Path, source: str) -> list[dict[str, object]]:
    """Load one source's immutable interpolation materializations."""
    path = root / "interpolation_materialization" / source / "fol_interpolation" / "records.jsonl"
    rows = read_jsonl(path)
    if not rows or any(row.get("source") != source or row.get("method") != "fol_interpolation" for row in rows):
        raise ValueError("interpolation materialization ledger is invalid")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--fol-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.source not in config.fol.sources:
        raise ValueError("interpolation generation requested an unconfigured source")
    root = args.fol_root if args.fol_root.is_absolute() else ROOT / args.fol_root
    rows = interpolation_materializations(root, args.source)
    model_path = config.models.surrogate.local_path
    if model_path is None:
        raise ValueError("interpolation generation requires a local target")
    resolved = validate_model_assets(model_path)
    handle = load_local_qwen(resolved, attention_backend=config.run.attention_implementation)
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("local interpolation target did not load")
        summary = generate_materialized_records(
            root, rows, model=handle.model, tokenizer=handle.tokenizer,
            target_key=config.models.targets[0].key, target_revision=resolved.revision,
            max_new_tokens=config.judging.max_new_tokens,
        )
    finally:
        handle.close()
    print(json.dumps({"failed": summary.failed_records, "selected": summary.selected_records, "written": summary.written_records}, sort_keys=True))
    return 0 if summary.failed_records == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
