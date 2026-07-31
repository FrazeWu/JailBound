"""Apply only the locked Octopus primary judge to H1-v2 response ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.safety_eval.config import load_h1_v2_config
from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.io import JsonlLedger, read_jsonl
from benchmark.safety_eval.h1_v2_runtime import h1_v2_eligible_records
from benchmark.safety_eval.judging import OctopusLocalJudge
from benchmark.safety_eval.pipeline import judge_response_records
from benchmark.safety_eval.runtime import validate_model_assets
from benchmark.safety_eval.schema import JudgmentRecord, ResponseRecord


ROOT = Path(__file__).resolve().parents[1]
METHOD = "fol_h1_v2"


def _selected_ids(root: Path, *, stage: str, source: str) -> set[str]:
    if stage == "baseline":
        examples, _ = h1_v2_eligible_records(root, source)
        return set(examples)
    if stage == "directions":
        return {
            str(row["perturbation_id"])
            for row in read_jsonl(root / "selected_directions.jsonl")
            if row.get("source") == source and isinstance(row.get("perturbation_id"), str)
        }
    raise ValueError("unknown H1-v2 judging stage")


def _responses(root: Path, *, target_key: str, sources: tuple[str, ...], stage: str) -> list[dict[str, object]]:
    records = []
    for source in sources:
        selected = _selected_ids(root, stage=stage, source=source)
        if not selected:
            raise ValueError(f"H1-v2 {stage} stage has no selected response IDs for {source}")
        rows = [
            ResponseRecord.model_validate(row)
            for row in read_jsonl(root / "responses" / target_key / source / METHOD / "records.jsonl")
        ]
        by_id = {row.sample_id: row for row in rows if row.checkpoint == 0 and row.sample_id in selected}
        if len(by_id) != len(selected) or set(by_id) != selected:
            raise ValueError(f"H1-v2 {stage} responses are incomplete for {source}")
        records.extend(by_id[record_id].model_dump(mode="json") for record_id in sorted(by_id))
    return records


def _write_numeric_labels(root: Path, *, target_key: str, sources: tuple[str, ...], stage: str, threshold: float) -> int:
    written = 0
    if stage == "baseline":
        ledger = JsonlLedger(root / "baseline_labels.jsonl", key_fields=("source", "sample_id"))
    else:
        ledger = JsonlLedger(root / "direction_labels.jsonl", key_fields=("perturbation_id",))
    for source in sources:
        selected = _selected_ids(root, stage=stage, source=source)
        path = root / "judgments" / "octopus_seval_14b" / target_key / source / METHOD / "records.jsonl"
        judgments = {
            row.sample_id: row
            for row in (JudgmentRecord.model_validate(item) for item in read_jsonl(path))
            if row.checkpoint == 0 and row.threshold == threshold and row.sample_id in selected
        }
        if set(judgments) != selected or any(row.status.value != "complete" for row in judgments.values()):
            raise ValueError(f"H1-v2 primary {stage} labels are incomplete for {source}")
        for record_id in sorted(selected):
            judgment = judgments[record_id]
            row = (
                {"source": source, "sample_id": record_id, "baseline_safe": not judgment.unsafe_label}
                if stage == "baseline"
                else {"perturbation_id": record_id, "unsafe_label": judgment.unsafe_label}
            )
            written += int(ledger.append_once(row))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("baseline", "directions"), required=True)
    parser.add_argument("--source", action="append")
    args = parser.parse_args()
    config = load_h1_v2_config(args.config)
    sources = tuple(args.source or config.h1_v2.sources)
    if set(sources) - set(config.h1_v2.sources):
        raise ValueError("H1-v2 judging requested an unconfigured source")
    if config.base.judging.primary.key != "octopus_seval_14b":
        raise ValueError("H1-v2 permits only the locked Octopus primary judge")
    root = ROOT / config.h1_v2.output_root
    target_key = config.base.models.targets[0].key
    responses = _responses(root, target_key=target_key, sources=sources, stage=args.stage)
    model_path = config.h1_v2.primary_judge_local_path
    resolved = validate_model_assets(model_path)
    handle = load_local_qwen(resolved)
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("H1-v2 local primary judge did not load")
        summary = judge_response_records(
            root,
            responses,
            judge=OctopusLocalJudge(model=handle.model, tokenizer=handle.tokenizer, revision=resolved.revision),
            threshold=config.base.judging.primary.threshold,
        )
    finally:
        handle.close()
    labels = _write_numeric_labels(
        root,
        target_key=target_key,
        sources=sources,
        stage=args.stage,
        threshold=config.base.judging.primary.threshold,
    )
    print(json.dumps({"failed": summary.failed_records, "labels": labels, "selected": summary.selected_records, "written": summary.written_records}, sort_keys=True))
    return 0 if summary.failed_records == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
