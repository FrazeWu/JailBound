"""Read the approved benchmark sources without changing their prompt text."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Callable


JAILBOUND_ATTACK_TYPES = frozenset(
    {
        "Persuasion & Deception",
        "Scenario Nesting",
        "Input Fragmentation",
        "Contextual Demonstration",
        "Obfuscation & Encryption",
        "Formal Language Exploitation",
        "Prefix Injection",
        "Code Injection",
        "Compositional / Hybrid Attacks",
    }
)
JAILBOUND_RISK_CATEGORIES = frozenset(
    {
        "Human-Chatbot Interaction Harms",
        "Human–Chatbot Interaction Harms",
        "Discrimination, Exclusion & Toxicity",
        "Sexual & Graphic Content",
        "Privacy & Personal Data",
        "Sensitive Organizational / Government Information",
        "Cybersecurity Misuse",
        "Illegal Wrongdoing & Criminal Enablement",
        "Fraud, Scams & Social Engineering",
        "Malicious Influence & Disinformation Campaigns",
        "Misinformation & Reliability Harms",
        "High-stakes Advice Harm",
        "Unsafe or Unethical Behavioral Encouragement",
    }
)
_HARM_BENCH_EXCLUDED = frozenset({"copyright"})
_SG_BENCH_CATEGORY = re.compile(r"\[\s*(O[1-6])\s*:", re.IGNORECASE)
_JAILBOUND_ATTACK = re.compile(r"Attack Type:\s*(.+?),\s*Risk Category:", re.DOTALL)
_JAILBOUND_RISK = re.compile(r"Risk Category:\s*(.+?),\s*and Attack Domain:", re.DOTALL)
_JAILBOUND_DOMAIN = re.compile(r"Attack Domain:\s*(.+?)(?:,\s*generate|\n|$)", re.DOTALL | re.IGNORECASE)


class SourceDataError(ValueError):
    """Raised when a source cannot meet its documented adapter contract."""


@dataclass(frozen=True)
class RawExample:
    source: str
    source_row: int
    intent: str
    attack_text: str
    target_text: str | None
    source_risk_label: str | None
    source_attack_label: str
    source_domain_label: str | None
    language: str
    preprocessing: tuple[str, ...]

    @property
    def source_row_id(self) -> str:
        """Stable source identity used by manifest construction and auditing."""
        attack_digest = hashlib.sha256(_normalize(self.attack_text).encode("utf-8")).hexdigest()
        return f"{self.source}:{self.source_row:06d}:{attack_digest[:12]}"


@dataclass(frozen=True)
class SourceLoadReport:
    source: str
    raw_count: int
    eligible_count: int
    exclusions: dict[str, int]


def _normalize(value: object) -> str:
    if not isinstance(value, str):
        raise SourceDataError("text value must be a string")
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _object_row(value: object, source: str, index: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SourceDataError(f"{source} row {index} must be an object")
    return value


def _raw(
    source: str,
    row: int,
    intent: object,
    attack_text: object,
    target_text: object | None,
    risk: object | None,
    attack: str,
    domain: object | None,
    *,
    language: str = "en",
    preprocessing: tuple[str, ...] = ("normalized_newlines", "trimmed_whitespace"),
) -> RawExample:
    return RawExample(
        source=source,
        source_row=row,
        intent=_normalize(intent),
        attack_text=_normalize(attack_text),
        target_text=None if target_text is None else _normalize(target_text),
        source_risk_label=None if risk is None else _normalize(risk),
        source_attack_label=attack,
        source_domain_label=None if domain is None else _normalize(domain),
        language=language,
        preprocessing=preprocessing,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _advbench(path: Path, _: Path | None) -> tuple[list[RawExample], SourceLoadReport]:
    rows = _read_csv(path)
    records = [_raw("advbench", index, row.get("goal"), row.get("goal"), row.get("target"), None, "direct_request", None) for index, row in enumerate(rows)]
    return records, SourceLoadReport("advbench", len(rows), len(records), {})


def _harmbench(path: Path, targets_path: Path | None) -> tuple[list[RawExample], SourceLoadReport]:
    if targets_path is None:
        raise SourceDataError("harmbench_targets_path is required for HarmBench")
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    if not isinstance(targets, dict):
        raise SourceDataError("HarmBench target file must contain an object")
    rows = _read_csv(path)
    excluded: dict[str, int] = {}
    records: list[RawExample] = []
    for index, row in enumerate(rows):
        functional_category = _normalize(row.get("FunctionalCategory", ""))
        if functional_category in _HARM_BENCH_EXCLUDED:
            excluded[functional_category] = excluded.get(functional_category, 0) + 1
            continue
        behavior_id = _normalize(row.get("BehaviorID", ""))
        if behavior_id not in targets:
            raise SourceDataError(f"HarmBench row {index} has no target for {behavior_id!r}")
        behavior = _normalize(row.get("Behavior", ""))
        context = _normalize(row.get("ContextString", ""))
        attack_text = f"{context}\n\n---\n\n{behavior}" if context else behavior
        records.append(_raw("harmbench", index, behavior, attack_text, targets[behavior_id], row.get("SemanticCategory"), "direct_request", None))
    return records, SourceLoadReport("harmbench", len(rows), len(records), excluded)


def _safetybench(path: Path, _: Path | None) -> tuple[list[RawExample], SourceLoadReport]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SourceDataError("SafetyBench source must contain a JSON list")
    records = []
    for index, row in enumerate(rows):
        row = _object_row(row, "SafetyBench", index)
        options = row.get("options", [])
        if not isinstance(options, list):
            raise SourceDataError(f"SafetyBench row {index} options must be a list")
        formatted_options = "\n".join(f"{chr(65 + option_index)}. {_normalize(option)}" for option_index, option in enumerate(options))
        attack_text = f"{_normalize(row.get('question', ''))}\nOptions:\n{formatted_options}"
        records.append(_raw("safetybench", index, row.get("question", ""), attack_text, None, row.get("category"), "direct_request", None))
    return records, SourceLoadReport("safetybench", len(rows), len(records), {})


def _sg_bench(path: Path, _: Path | None) -> tuple[list[RawExample], SourceLoadReport]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SourceDataError("SG-Bench source must contain a JSON list")
    records = []
    for index, row in enumerate(rows):
        row = _object_row(row, "SG-Bench", index)
        match = _SG_BENCH_CATEGORY.search(_normalize(row.get("safety_type", "")))
        if match is None:
            raise SourceDataError(f"SG-Bench row {index} has no O1-O6 label")
        responses = row.get("reference_responses")
        if responses is None:
            target_text = None
        elif not isinstance(responses, list):
            raise SourceDataError(f"SG-Bench row {index} reference_responses must be a list")
        else:
            target_text = responses[0] if responses else None
        records.append(_raw("sg_bench", index, row.get("query", ""), row.get("query", ""), target_text, match.group(1).upper(), "direct_request", None))
    return records, SourceLoadReport("sg_bench", len(rows), len(records), {})


def _jailbreakbench(path: Path, _: Path | None) -> tuple[list[RawExample], SourceLoadReport]:
    rows = _read_csv(path)
    records = [_raw("jailbreakbench", index, row.get("Goal"), row.get("Goal"), row.get("Target"), row.get("Category"), "direct_request", None) for index, row in enumerate(rows)]
    return records, SourceLoadReport("jailbreakbench", len(rows), len(records), {})


def _match(pattern: re.Pattern[str], instruction: str, field: str, index: int) -> str:
    match = pattern.search(instruction)
    if match is None or not _normalize(match.group(1)):
        raise SourceDataError(f"JailBound row {index} is missing {field}")
    return _normalize(match.group(1))


def _jailbound(path: Path, _: Path | None) -> tuple[list[RawExample], SourceLoadReport]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SourceDataError("JailBound source must contain a JSON list")
    records = []
    for index, row in enumerate(rows):
        row = _object_row(row, "JailBound", index)
        instruction = _normalize(row.get("instruction", ""))
        attack = _match(_JAILBOUND_ATTACK, instruction, "Attack Type", index)
        risk = _match(_JAILBOUND_RISK, instruction, "Risk Category", index)
        domain = _match(_JAILBOUND_DOMAIN, instruction, "Attack Domain", index)
        if attack not in JAILBOUND_ATTACK_TYPES or risk not in JAILBOUND_RISK_CATEGORIES:
            raise SourceDataError(f"JailBound row {index} has an unrecognized native label")
        records.append(_raw("jailbound", index, row.get("input", ""), row.get("output", ""), None, risk, attack, domain))
    return records, SourceLoadReport("jailbound", len(rows), len(records), {})


def _s_eval(path: Path, _: Path | None) -> tuple[list[RawExample], SourceLoadReport]:
    if "_en_" not in path.name.lower():
        raise SourceDataError("S-Eval adapter requires the English source file")
    records: list[RawExample] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            row = _object_row(row, "S-Eval", index)
            try:
                ext = json.loads(row["ext"])
                attack = ext["category"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise SourceDataError(f"S-Eval row {index} has invalid ext JSON") from error
            records.append(_raw("s_eval", index, row.get("prompt", ""), row.get("prompt", ""), None, row.get("risk_type"), _normalize(attack), None, language="en"))
    return records, SourceLoadReport("s_eval", len(records), len(records), {})


_ADAPTERS: dict[str, Callable[[Path, Path | None], tuple[list[RawExample], SourceLoadReport]]] = {
    "advbench": _advbench,
    "harmbench": _harmbench,
    "safetybench": _safetybench,
    "sg_bench": _sg_bench,
    "jailbreakbench": _jailbreakbench,
    "jailbound": _jailbound,
    "s_eval": _s_eval,
}


def load_source_with_report(source: str, path: Path, harmbench_targets_path: Path | None = None) -> tuple[list[RawExample], SourceLoadReport]:
    try:
        adapter = _ADAPTERS[source]
    except KeyError as error:
        raise SourceDataError(f"unsupported benchmark source: {source}") from error
    return adapter(Path(path), None if harmbench_targets_path is None else Path(harmbench_targets_path))


def load_source(source: str, path: Path, harmbench_targets_path: Path | None = None) -> list[RawExample]:
    """Load a source through its exact adapter contract."""
    records, _ = load_source_with_report(source, path, harmbench_targets_path)
    return records
