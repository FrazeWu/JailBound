"""Atomic output writing for Alpaca annotation artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import AlpacaRecord, DetailRecord, Language


LANGUAGE_FILES: dict[Language, str] = {
    "en": "alpaca_en.json",
    "zh": "alpaca_zh.json",
    "multilingual": "alpaca_multilingual.json",
    "unknown": "alpaca_unknown.json",
}


def write_outputs(output_dir: Path, details: list[DetailRecord], manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_language: dict[Language, list[AlpacaRecord]] = {
        "en": [],
        "zh": [],
        "multilingual": [],
        "unknown": [],
    }
    for detail in details:
        by_language[detail.language].append(
            AlpacaRecord(
                instruction=detail.instruction,
                input=detail.input,
                output=detail.output,
            )
        )

    for language, filename in LANGUAGE_FILES.items():
        _atomic_write_json(
            output_dir / filename,
            [record.to_dict() for record in by_language[language]],
        )
    _atomic_write_jsonl(output_dir / "alpaca_detail.jsonl", [detail.to_dict() for detail in details])
    _atomic_write_json(output_dir / "manifest.json", manifest)


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
