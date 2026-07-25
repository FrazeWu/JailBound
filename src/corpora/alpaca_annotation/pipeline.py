"""Pipeline orchestration for downloaded dataset Alpaca annotation."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .extractors import deduplicate_records, extract_records_from_file, scan_candidate_files
from .language import detect_language
from .models import Annotation, DetailRecord, ExtractedRecord, Language
from .taxonomy import build_instruction
from .writer import write_outputs


class Annotator(Protocol):
    def annotate_one(self, record: ExtractedRecord) -> Annotation:
        ...


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    limit: int | None = None
    language_quotas: dict[Language, int] | None = None
    datasets: set[str] | None = None
    scan_only: bool = False
    force: bool = False
    workers: int = 1


def run_pipeline(config: PipelineConfig, annotator: Annotator | None = None) -> dict:
    candidates, scan_decisions = scan_candidate_files(config.input_dir)
    if config.datasets:
        candidates = [path for path in candidates if path.relative_to(config.input_dir).parts[0] in config.datasets]
        scan_decisions = [decision for decision in scan_decisions if decision.source_dataset in config.datasets]

    extracted: list[ExtractedRecord] = []
    decisions = list(scan_decisions)
    for path in candidates:
        records, decision = extract_records_from_file(path, config.input_dir)
        decisions.append(decision)
        extracted.extend(records)

    unique, duplicate_count = deduplicate_records(extracted)
    if config.limit is not None:
        unique = unique[: config.limit]
    selected_languages: dict[str, Language] = {}
    language_quota_selected: dict[Language, int] | None = None
    if config.language_quotas:
        unique, language_quota_selected, selected_languages = _apply_language_quotas(unique, config.language_quotas)

    checkpoint_path = config.output_dir / ".checkpoint.jsonl"
    if config.force and checkpoint_path.exists():
        checkpoint_path.unlink()
    checkpoint = {} if config.force or config.scan_only else _load_checkpoint(checkpoint_path)

    details: list[DetailRecord] = []
    annotation_errors: list[dict] = []
    pending: list[ExtractedRecord] = []
    for record in unique:
        saved = checkpoint.get(record.id)
        if saved:
            details.append(_detail_from_checkpoint(saved))
        elif not config.scan_only:
            pending.append(record)
    if pending:
        if annotator is None:
            raise ValueError("annotator is required unless scan_only is true")
        for detail, error in _annotate_records(pending, annotator, selected_languages, max(config.workers, 1)):
            if error is not None:
                annotation_errors.append(error)
                continue
            if detail is None:
                continue
            details.append(detail)
            _append_checkpoint(checkpoint_path, detail)

    manifest = {
        "summary": {
            "candidate_files": len(candidates),
            "records_extracted": len(extracted),
            "duplicates_removed": duplicate_count,
            "records_after_limit": len(unique),
            "converted_records": len(details),
            "annotation_errors": len(annotation_errors),
        },
        "file_decisions": [decision.to_dict() for decision in decisions],
        "annotation_errors_detail": annotation_errors,
    }
    if config.language_quotas:
        manifest["summary"]["records_after_language_quota"] = len(unique)
        manifest["summary"]["language_quota_targets"] = dict(config.language_quotas)
        manifest["summary"]["language_quota_selected"] = dict(language_quota_selected or {})
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not config.scan_only:
        write_outputs(config.output_dir, details, manifest)
    else:
        (config.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return manifest


def _annotate_records(
    records: list[ExtractedRecord],
    annotator: Annotator,
    selected_languages: dict[str, Language],
    workers: int,
):
    if workers == 1:
        for record in records:
            yield _annotate_record(record, annotator, selected_languages)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_annotate_record, record, annotator, selected_languages): record
            for record in records
        }
        for future in as_completed(futures):
            yield future.result()


def _annotate_record(
    record: ExtractedRecord,
    annotator: Annotator,
    selected_languages: dict[str, Language],
) -> tuple[DetailRecord | None, dict | None]:
    try:
        annotation = annotator.annotate_one(record)
    except Exception as exc:
        return None, {"id": record.id, "error": str(exc)}
    language = selected_languages.get(record.id) or detect_language(record.prompt, annotation.malicious_intent)
    detail = DetailRecord(
        id=record.id,
        source_dataset=record.source_dataset,
        source_file=record.source_file,
        source_row=record.source_row,
        language=language,
        prompt_hash=record.prompt_hash,
        instruction=build_instruction(annotation.attack_type, annotation.risk_category, annotation.domain),
        input=annotation.malicious_intent,
        output=record.prompt,
        attack_type=annotation.attack_type,
        risk_category=annotation.risk_category,
        domain=annotation.domain,
        raw_annotation=annotation.raw,
        conversion_notes=record.notes,
    )
    return detail, None


def _apply_language_quotas(
    records: list[ExtractedRecord],
    language_quotas: dict[Language, int],
) -> tuple[list[ExtractedRecord], dict[Language, int], dict[str, Language]]:
    selected: list[ExtractedRecord] = []
    selected_languages: dict[str, Language] = {}
    counts: dict[Language, int] = {language: 0 for language in language_quotas}
    for record in records:
        language = detect_language(record.prompt)
        quota = language_quotas.get(language)
        if quota is None or counts[language] >= quota:
            continue
        selected.append(record)
        selected_languages[record.id] = language
        counts[language] += 1
        if all(counts[language] >= quota for language, quota in language_quotas.items()):
            break
    return selected, counts, selected_languages


def _load_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[data["id"]] = data
    return done


def _append_checkpoint(path: Path, detail: DetailRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(detail.to_dict(), ensure_ascii=False) + "\n")


def _detail_from_checkpoint(data: dict) -> DetailRecord:
    return DetailRecord(**data)
