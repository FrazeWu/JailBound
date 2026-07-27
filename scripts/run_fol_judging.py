"""Judge persisted FOL responses with one configured local judge without printing text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.reviewer_eval.config import load_config
from benchmark.reviewer_eval.execution import load_local_qwen
from benchmark.reviewer_eval.io import read_jsonl
from benchmark.reviewer_eval.judging import OctopusLocalJudge, Qwen32CompatJudge
from benchmark.reviewer_eval.pipeline import judge_response_records
from benchmark.reviewer_eval.runtime import validate_model_assets
from benchmark.reviewer_eval.schema import ResponseRecord


ROOT = Path(__file__).resolve().parents[1]


def _responses(
    root: Path, *, target_key: str, sources: tuple[str, ...], method: str
) -> list[dict[str, object]]:
    if method == "fol_boundary":
        metadata_path, id_field = root / "selected_perturbations.jsonl", "perturbation_id"
    elif method == "fol_interpolation":
        metadata_path, id_field = root / "interpolation_points.jsonl", "interpolation_id"
    else:
        raise ValueError("unknown FOL judging method")
    metadata = [row for row in read_jsonl(metadata_path) if row.get("source") in sources]
    ids_by_source: dict[str, set[str]] = {source: set() for source in sources}
    for row in metadata:
        source, record_id = row.get("source"), row.get(id_field)
        if not isinstance(source, str) or not isinstance(record_id, str):
            raise ValueError("FOL judging metadata is invalid")
        ids_by_source[source].add(record_id)
    records: list[dict[str, object]] = []
    for source in sources:
        rows = [
            ResponseRecord.model_validate(row)
            for row in read_jsonl(root / "responses" / target_key / source / method / "records.jsonl")
        ]
        selected = [row for row in rows if row.checkpoint == 0] if method == "fol_boundary" else rows
        by_id = {row.sample_id: row for row in selected}
        if len(by_id) != len(selected) or set(by_id) != ids_by_source[source]:
            raise ValueError(f"FOL responses are incomplete for {source}/{method}")
        records.extend(by_id[record_id].model_dump(mode="json") for record_id in sorted(by_id))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--judge", choices=("primary", "secondary"), required=True)
    parser.add_argument("--method", choices=("fol_boundary", "fol_interpolation"), default="fol_boundary")
    parser.add_argument("--source", action="append")
    args = parser.parse_args()
    config = load_config(args.config)
    sources = tuple(args.source or config.fol.sources)
    if set(sources) - set(config.fol.sources):
        raise ValueError("FOL judging requested an unconfigured source")
    root = ROOT / config.run.output_root / "fol_boundary"
    target_key = config.models.targets[0].key
    responses = _responses(root, target_key=target_key, sources=sources, method=args.method)
    if args.judge == "primary":
        model_path = config.models.octopus.local_path
        if model_path is None:
            raise ValueError("FOL primary judge requires a local Octopus model")
        resolved = validate_model_assets(model_path)
        handle = load_local_qwen(resolved)
        try:
            if handle.model is None or handle.tokenizer is None:
                raise ValueError("local FOL primary judge did not load")
            judge = OctopusLocalJudge(model=handle.model, tokenizer=handle.tokenizer, revision=resolved.revision)
            summary = judge_response_records(root, responses, judge=judge, threshold=config.judging.primary.threshold)
        finally:
            handle.close()
    else:
        spec = config.judging.secondary
        if spec.endpoint is None or spec.model is None:
            raise ValueError("FOL secondary judge endpoint and model are required")
        with Qwen32CompatJudge(
            endpoint=spec.endpoint,
            model=spec.model,
            max_new_tokens=config.judging.max_new_tokens,
        ) as judge:
            summary = judge_response_records(root, responses, judge=judge, threshold=spec.threshold)
    print(json.dumps({
        "failed": summary.failed_records,
        "judge": args.judge,
        "selected": summary.selected_records,
        "written": summary.written_records,
    }, sort_keys=True))
    return 0 if summary.failed_records == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
