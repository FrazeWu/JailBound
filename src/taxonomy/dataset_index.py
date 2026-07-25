#!/usr/bin/env python3
"""
dataset_index.py — Unified dataset loader for the benchmark framework.

Provides a single interface to load harmful-prompt datasets organised under
``corpora/``.  Callers can filter by:

    - threat_category : e.g. "cybersecurity_misuse", "illegal_criminal", …  (12 snake_case keys)
    - attack_type     : e.g. "scenario_nesting", "persuasion_deception", … (8 keys)
    - source          : e.g. "harmbench", "advbench", "strongreject", "s_eval"

Usage
-----
>>> from dataset_index import DatasetIndex
>>> idx = DatasetIndex()
>>> records = idx.load(threat_category="cybersecurity_misuse", attack_type="prefix_code_injection")
>>> print(len(records), records[0].keys())

Each record has the schema:
    {
      "id":              str,
      "behavior":        str,
      "target":          str | None,
      "threat_category": str,          # snake_case key
      "attack_type":     str,          # snake_case key
      "source":          str,
      "tags":            list[str],
    }

Data sources loaded (in priority order, deduped by behavior prefix):
    1. attack_types/{attack_type}/samples.jsonl   (pre-partitioned)
    2. threat_categories/{threat_key}/samples.jsonl
    3. benchmark/data/harmbench_behaviors.jsonl  (master combined file)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
BENCHMARK_DATA = _HERE.parent / "benchmark" / "data" / "harmbench_behaviors.jsonl"

# 12 threat category snake_case keys
THREAT_CATEGORIES: set[str] = {
    "human_chatbot_harm",
    "discrimination_toxicity",
    "sexual_graphic",
    "privacy_personal_data",
    "sensitive_org_gov",
    "cybersecurity_misuse",
    "illegal_criminal",
    "fraud_scam",
    "malicious_influence",
    "misinformation_reliability",
    "high_stakes_advice",
    "unsafe_unethical",
}

# 8 attack type keys (snake_case)
ATTACK_TYPES: set[str] = {
    "persuasion_deception",
    "scenario_nesting",
    "input_fragmentation",
    "contextual_demonstration",
    "obfuscation_encryption",
    "formal_language",
    "prefix_code_injection",
    "compositional_hybrid",
}

# attack_types directory → sub-directory names
ATTACK_TYPE_DIRS: dict[str, str] = {
    "persuasion_deception": "persuasion_deception",
    "scenario_nesting": "scenario_nesting",
    "input_fragmentation": "input_fragmentation",
    "contextual_demonstration": "contextual_demonstration",
    "obfuscation_encryption": "obfuscation_encryption",
    "formal_language": "formal_language",
    "prefix_code_injection": "prefix_code_injection",
    "compositional_hybrid": "compositional_hybrid",
}

# threat_categories directory → sub-directory names (keys and values are identical snake_case)
THREAT_CATEGORY_DIRS: dict[str, str] = {
    "discrimination_toxicity": "discrimination_toxicity",
    "sexual_graphic": "sexual_graphic",
    "privacy_personal_data": "privacy_personal_data",
    "sensitive_org_gov": "sensitive_org_gov",
    "cybersecurity_misuse": "cybersecurity_misuse",
    "illegal_criminal": "illegal_criminal",
    "fraud_scam": "fraud_scam",
    "malicious_influence": "malicious_influence",
    "misinformation_reliability": "misinformation_reliability",
    "high_stakes_advice": "high_stakes_advice",
    "unsafe_unethical": "unsafe_unethical",
    "human_chatbot_harm": "human_chatbot_harm",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_threat(value: str | None) -> str | None:
    """Validate that value is a known snake_case threat category key, or None."""
    if value is None:
        return None
    if value in THREAT_CATEGORIES:
        return value
    return None


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _dedup_key(record: dict) -> str:
    behavior = record.get("behavior", "")
    return re.sub(r"\s+", " ", behavior[:80].lower().strip())


def _dedup(records: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        k = _dedup_key(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DatasetIndex:
    """Lazy-loading index over all benchmark datasets.

    Args:
        root: Root directory (defaults to this file's parent directory).
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root) if root else _HERE
        self._benchmark_data = (
            self._root.parent / "benchmark" / "data" / "harmbench_behaviors.jsonl"
        )
        self._attack_types_dir = self._root / "attack_types"
        self._threat_cats_dir = self._root / "threat_categories"
        self._cache: dict[str, list[dict]] | None = None

    # ------------------------------------------------------------------
    # Core load
    # ------------------------------------------------------------------

    def _load_all(self) -> list[dict]:
        """Load and deduplicate all available data sources."""
        if self._cache is not None:
            return self._cache

        records: list[dict] = []

        # 1. Per-attack_type JSONL files (highest priority — curated)
        for attack_key, dirname in ATTACK_TYPE_DIRS.items():
            samples_path = self._attack_types_dir / dirname / "samples.jsonl"
            for r in _load_jsonl(samples_path):
                r.setdefault("attack_type", attack_key)
                r.setdefault("source", "curated")
                records.append(r)

        # 2. Per-threat_category JSONL files
        for snake_key, dirname in THREAT_CATEGORY_DIRS.items():
            samples_path = self._threat_cats_dir / dirname / "samples.jsonl"
            for r in _load_jsonl(samples_path):
                r.setdefault("threat_category", snake_key)
                r.setdefault("source", "curated")
                records.append(r)

        # 3. Master combined file (harmbench_behaviors.jsonl)
        records.extend(_load_jsonl(self._benchmark_data))

        self._cache = _dedup(records)
        return self._cache

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    def load(
        self,
        threat_category: str | None = None,
        attack_type: str | None = None,
        source: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Return records matching all provided filters.

        Args:
            threat_category: Snake-case key (``"cybersecurity_misuse"``).
                ``None`` = no filter.
            attack_type: Snake-case key, e.g. ``"scenario_nesting"``.
                ``None`` = no filter.
            source: One of ``"harmbench"``, ``"advbench"``,
                ``"strongreject"``, ``"s_eval"``, ``"curated"``.
                ``None`` = no filter.
            limit: Maximum number of records to return.

        Returns:
            List of matching record dicts.
        """
        all_records = self._load_all()
        threat_abbr = _normalise_threat(threat_category)

        out: list[dict] = []
        for r in all_records:
            if threat_abbr and r.get("threat_category") != threat_abbr:
                continue
            if attack_type and r.get("attack_type") != attack_type:
                continue
            if source and r.get("source") != source:
                continue
            out.append(r)
            if limit and len(out) >= limit:
                break

        return out

    def load_all(self) -> list[dict]:
        """Return all deduplicated records."""
        return list(self._load_all())

    def count(
        self,
        threat_category: str | None = None,
        attack_type: str | None = None,
        source: str | None = None,
    ) -> int:
        """Count records matching the given filters."""
        return len(
            self.load(
                threat_category=threat_category, attack_type=attack_type, source=source
            )
        )

    def summary(self) -> dict:
        """Return a summary dict with counts by threat_category, attack_type, source."""
        from collections import Counter

        all_records = self._load_all()
        return {
            "total": len(all_records),
            "by_threat_category": dict(
                Counter(r.get("threat_category", "?") for r in all_records)
            ),
            "by_attack_type": dict(
                Counter(r.get("attack_type", "?") for r in all_records)
            ),
            "by_source": dict(Counter(r.get("source", "?") for r in all_records)),
        }

    def get_behaviors_for_benchmark(
        self,
        threat_category: str | None = None,
        attack_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Convenience method: return records in benchmark-compatible format.

        Always includes ``behavior``, ``target``, ``threat_category``,
        ``attack_type``, ``source`` keys.
        """
        records = self.load(
            threat_category=threat_category,
            attack_type=attack_type,
            limit=limit,
        )
        out = []
        for r in records:
            out.append(
                {
                    "id": r.get("id", ""),
                    "behavior": r.get("behavior", ""),
                    "target": r.get("target", ""),
                    "threat_category": r.get("threat_category", ""),
                    "attack_type": r.get("attack_type", ""),
                    "source": r.get("source", ""),
                    "tags": r.get("tags", []),
                }
            )
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Query the benchmark dataset index.")
    parser.add_argument(
        "--threat", default=None, help="Threat category snake_case key (e.g. cybersecurity_misuse)"
    )
    parser.add_argument("--attack", default=None, help="Attack type key (snake_case)")
    parser.add_argument("--source", default=None, help="Source name")
    parser.add_argument("--limit", type=int, default=None, help="Max records to print")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary")
    args = parser.parse_args()

    idx = DatasetIndex()

    if args.summary:
        s = idx.summary()
        print(f"Total records: {s['total']}\n")
        print("By threat_category:")
        for k, v in sorted(s["by_threat_category"].items()):
            print(f"  {k:6s}: {v}")
        print("\nBy attack_type:")
        for k, v in sorted(s["by_attack_type"].items()):
            print(f"  {k:32s}: {v}")
        print("\nBy source:")
        for k, v in sorted(s["by_source"].items()):
            print(f"  {k:15s}: {v}")
        return

    records = idx.load(
        threat_category=args.threat,
        attack_type=args.attack,
        source=args.source,
        limit=args.limit,
    )
    print(f"Found {len(records)} records")
    for r in records[:10]:
        print(json.dumps(r, ensure_ascii=False))
    if len(records) > 10:
        print(f"... ({len(records) - 10} more)")


if __name__ == "__main__":
    _cli()
