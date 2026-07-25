"""Shared data models for Alpaca dataset annotation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Language = Literal["en", "zh", "multilingual", "unknown"]


@dataclass(frozen=True)
class ExtractedRecord:
    id: str
    source_dataset: str
    source_file: str
    source_row: int
    prompt: str
    prompt_hash: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Annotation:
    attack_type: str
    risk_category: str
    domain: str
    malicious_intent: str
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlpacaRecord:
    instruction: str
    input: str
    output: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DetailRecord:
    id: str
    source_dataset: str
    source_file: str
    source_row: int
    language: Language
    prompt_hash: str
    instruction: str
    input: str
    output: str
    attack_type: str
    risk_category: str
    domain: str
    raw_annotation: dict[str, Any]
    conversion_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FileDecision:
    path: str
    source_dataset: str
    status: str
    reason: str
    records_seen: int = 0
    records_extracted: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
