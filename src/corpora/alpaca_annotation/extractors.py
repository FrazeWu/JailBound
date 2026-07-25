"""Conservative source discovery and prompt extraction."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .models import ExtractedRecord, FileDecision


DATA_EXTENSIONS = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".parquet"}
NON_DATA_PARTS = {
    ".git",
    "__pycache__",
    ".cache",
    "node_modules",
    "venv",
    ".venv",
    "docs",
    "doc",
    "scripts",
    "script",
    "results",
    "result",
    "logs",
    "log",
    "assets",
    "asset",
    "images",
    "image",
    "img",
    "notebooks",
    "notebook",
    "checkpoints",
    "checkpoint",
}
README_NAMES = {"readme", "license", "citation", "requirements", "setup", "pyproject"}
PROMPT_FIELDS = [
    "prompt",
    "jailbreak_prompt",
    "attack_prompt",
    "adversarial_prompt",
    "behavior",
    "goal",
    "question",
    "instruction",
    "query",
    "text",
]
ADAPTER_FIELDS: dict[str, list[str]] = {
    "AdvBench": ["goal"],
    "StrongREJECT": ["forbidden_prompt", "prompt"],
    "HarmBench": ["Behavior", "behavior", "prompt"],
    "JBBBehaviours": ["Goal", "goal", "behavior"],
    "SafetyPrompts": ["prompt"],
    "DoNotAnswer": ["question", "prompt"],
    "XSTest": ["prompt"],
    "SorryBench": ["prompt", "question"],
    "BeaverTails": ["prompt"],
    "Aegis-AI-Content-Safety-Dataset-2.0": ["prompt", "text"],
    "AegisAIContentSafety": ["prompt", "text"],
}


def scan_candidate_files(root: Path) -> tuple[list[Path], list[FileDecision]]:
    candidates: list[Path] = []
    decisions: list[FileDecision] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        source_dataset = _source_dataset(path, root)
        rel = path.relative_to(root).as_posix()
        if _is_non_data_path(path, root):
            decisions.append(FileDecision(rel, source_dataset, "skipped_non_data_path", "known non-data path"))
            continue
        if path.suffix.lower() not in DATA_EXTENSIONS:
            decisions.append(FileDecision(rel, source_dataset, "skipped_unsupported_extension", path.suffix.lower() or "no extension"))
            continue
        candidates.append(path)
    return candidates, decisions


def extract_records_from_file(path: Path, root: Path) -> tuple[list[ExtractedRecord], FileDecision]:
    source_dataset = _source_dataset(path, root)
    rel = path.relative_to(root).as_posix()
    try:
        rows = _load_rows(path)
    except Exception as exc:
        return [], FileDecision(rel, source_dataset, "skipped_parse_error", str(exc))
    if not rows:
        return [], FileDecision(rel, source_dataset, "skipped_empty", "no usable rows")

    records: list[ExtractedRecord] = []
    ambiguous = 0
    for idx, row in enumerate(rows):
        prompt, note = _extract_prompt(row, source_dataset)
        if prompt is None:
            if note.startswith("ambiguous"):
                ambiguous += 1
            continue
        prompt_hash = _prompt_hash(prompt)
        stable_id = f"{source_dataset}/{rel}/{idx}/{prompt_hash}"
        records.append(
            ExtractedRecord(
                id=stable_id,
                source_dataset=source_dataset,
                source_file=rel,
                source_row=idx,
                prompt=prompt,
                prompt_hash=prompt_hash,
                notes=[note] if note else [],
            )
        )

    if not records:
        reason = "ambiguous prompt fields" if ambiguous else "no reliable prompt field"
        return [], FileDecision(rel, source_dataset, "skipped_no_prompt_field", reason, records_seen=len(rows))

    return records, FileDecision(
        rel,
        source_dataset,
        "converted",
        f"{len(records)} records extracted",
        records_seen=len(rows),
        records_extracted=len(records),
    )


def deduplicate_records(records: list[ExtractedRecord]) -> tuple[list[ExtractedRecord], int]:
    seen: set[str] = set()
    unique: list[ExtractedRecord] = []
    duplicates = 0
    for record in records:
        if record.prompt_hash in seen:
            duplicates += 1
            continue
        seen.add(record.prompt_hash)
        unique.append(record)
    return unique, duplicates


def _load_rows(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return _flatten_json_dict(data)
        return []
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter=delimiter))
    if suffix == ".txt":
        return [{"text": line.strip()} for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    return []


def _flatten_json_dict(data: dict[str, Any]) -> list[Any]:
    if all(isinstance(value, list) for value in data.values()):
        if any(any(isinstance(item, dict) for item in values) for values in data.values()):
            rows: list[Any] = []
            for key, values in data.items():
                for value in values:
                    if isinstance(value, dict):
                        rows.append({"_group": key, **value})
            return rows

        lengths = {len(values) for values in data.values()}
        if len(lengths) == 1:
            row_count = lengths.pop()
            return [
                {key: values[idx] for key, values in data.items()}
                for idx in range(row_count)
            ]

        rows: list[Any] = []
        for key, values in data.items():
            for value in values:
                rows.append({key: value})
        return rows
    return [data]


def _extract_prompt(row: Any, source_dataset: str) -> tuple[str | None, str]:
    if isinstance(row, str):
        text = row.strip()
        return (text, "plain text row") if text else (None, "empty string")
    if not isinstance(row, dict):
        return None, "unsupported row type"

    if source_dataset in ADAPTER_FIELDS:
        for field in ADAPTER_FIELDS[source_dataset]:
            if isinstance(row.get(field), str) and row.get(field, "").strip():
                return row[field].strip(), f"field:{field}"

    generic_present = [
        field
        for field in PROMPT_FIELDS
        if isinstance(row.get(field), str) and row.get(field, "").strip()
    ]
    if len(generic_present) == 1:
        field = generic_present[0]
        return row[field].strip(), f"field:{field}"
    if len(generic_present) > 1:
        return None, f"ambiguous fields:{','.join(generic_present)}"
    return None, "no prompt field"


def _source_dataset(path: Path, root: Path) -> str:
    rel_parts = path.relative_to(root).parts
    return rel_parts[0] if rel_parts else "unknown"


def _is_non_data_path(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    lowered = {part.lower() for part in rel_parts[:-1]}
    if lowered & NON_DATA_PARTS:
        return True
    stem = path.stem.lower()
    return stem in README_NAMES


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
