"""Apply only the locked Octopus primary judge to H1-v3 response ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.safety_eval.config import load_h1_v3_config
from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.h1_v3_runtime import H1V3Paths
from benchmark.safety_eval.io import JsonlLedger, read_jsonl
from benchmark.safety_eval.judging import OctopusLocalJudge
from benchmark.safety_eval.pipeline import judge_response_records
from benchmark.safety_eval.runtime import validate_model_assets
from benchmark.safety_eval.schema import JudgmentRecord, ResponseRecord


ROOT = Path(__file__).resolve().parents[1]
METHOD = "fol_h1_v3"


def _responses(paths: H1V3Paths, *, target_key: str, sources: tuple[str, ...]) -> list[dict[str, object]]:
    records = []
    directions = read_jsonl(paths.selected_directions())
    for source in sources:
        selected = {
            str(row["perturbation_id"])
            for row in directions
            if row.get("source") == source and isinstance(row.get("perturbation_id"), str)
        }
        if not selected:
            raise ValueError(f"H1-v3 selected directions are empty for {source}")
        rows = [
            ResponseRecord.model_validate(row)
            for row in read_jsonl(paths.response_ledger(target_key, source))
        ]
        by_id = {
            row.sample_id: row
            for row in rows
            if row.checkpoint == 0 and row.sample_id in selected
        }
        if set(by_id) != selected:
            raise ValueError(f"H1-v3 responses are incomplete for {source}")
        records.extend(by_id[record_id].model_dump(mode="json") for record_id in sorted(by_id))
    return records


def _write_numeric_labels(paths: H1V3Paths, *, target_key: str, sources: tuple[str, ...], threshold: float) -> int:
    written = 0
    ledger = JsonlLedger(paths.direction_labels(), key_fields=("perturbation_id",))
    directions = read_jsonl(paths.selected_directions())
    for source in sources:
        selected = {
            str(row["perturbation_id"])
            for row in directions
            if row.get("source") == source and isinstance(row.get("perturbation_id"), str)
        }
        judgments = {
            row.sample_id: row
            for row in (
                JudgmentRecord.model_validate(item)
                for item in read_jsonl(paths.judgment_ledger(target_key, source))
            )
            if row.checkpoint == 0 and row.threshold == threshold and row.sample_id in selected
        }
        if set(judgments) != selected or any(row.status.value != "complete" for row in judgments.values()):
            raise ValueError(f"H1-v3 primary labels are incomplete for {source}")
        for perturbation_id in sorted(selected):
            written += int(ledger.append_once({
                "perturbation_id": perturbation_id,
                "unsafe_label": judgments[perturbation_id].unsafe_label,
            }))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", action="append")
    args = parser.parse_args()
    config = load_h1_v3_config(args.config)
    sources = tuple(args.source or config.h1_v3.sources)
    if set(sources) - set(config.h1_v3.sources):
        raise ValueError("H1-v3 judging requested an unconfigured source")
    if config.base.judging.primary.key != "octopus_seval_14b":
        raise ValueError("H1-v3 permits only the locked Octopus primary judge")
    paths = H1V3Paths(ROOT / config.h1_v3.source_root, ROOT / config.h1_v3.output_root)
    target_key = config.base.models.targets[0].key
    responses = _responses(paths, target_key=target_key, sources=sources)
    resolved = validate_model_assets(config.h1_v3.primary_judge_local_path)
    handle = load_local_qwen(resolved, attention_backend=config.base.run.attention_implementation)
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("H1-v3 local primary judge did not load")
        summary = judge_response_records(
            paths.output_root,
            responses,
            judge=OctopusLocalJudge(model=handle.model, tokenizer=handle.tokenizer, revision=resolved.revision),
            threshold=config.base.judging.primary.threshold,
        )
    finally:
        handle.close()
    labels = _write_numeric_labels(
        paths,
        target_key=target_key,
        sources=sources,
        threshold=config.base.judging.primary.threshold,
    )
    print(json.dumps({
        "failed": summary.failed_records,
        "labels": labels,
        "selected": summary.selected_records,
        "written": summary.written_records,
    }, sort_keys=True))
    return 0 if summary.failed_records == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
