"""
QuoTe Dataset Loader Module

Loads HarmBench behavior datasets and maps them to our threat taxonomy.
Provides a unified list of behavior dicts for the compliance boundary detection pipeline.

Dataset columns (from harmbench_behaviors_text_all.csv):
    Behavior, FunctionalCategory, SemanticCategory, Tags, ContextString, BehaviorID

Output behavior dicts have the schema:
    {
        "behavior_id": str,           # HarmBench BehaviorID
        "behavior": str,              # The behavior string (harmful instruction)
        "functional_category": str,   # "standard" | "contextual"
        "semantic_category": str,     # HarmBench SemanticCategory
        "context_string": str,        # Context (for contextual behaviors; may be "")
        "tags": str,                  # Raw tags string
        "threat_categories": list[str],  # Mapped threat IDs from our taxonomy
        "primary_threat": str | None, # First threat ID, or None
        "source": str,                # "harmbench"
    }
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path defaults — relative to project root
HARMBENCH_TEXT_ALL = "data/benchmark/harmbench_behaviors.jsonl"
HARMBENCH_TEXT_TEST = "data/benchmark/harmbench_behaviors_test.jsonl"

# Canonical HarmBench CSV column names
COL_BEHAVIOR = "Behavior"
COL_FUNCTIONAL = "FunctionalCategory"
COL_SEMANTIC = "SemanticCategory"
COL_TAGS = "Tags"
COL_CONTEXT = "ContextString"
COL_BEHAVIOR_ID = "BehaviorID"


# =============================================================================
# HarmBench → threat taxonomy mapping (inline for zero extra import)
# The canonical version lives in attack_model/taxonomy/harmbench_mapping.py;
# we duplicate here to keep quote self-contained.
# =============================================================================

_HARMBENCH_TO_THREAT: dict[str, list[str]] = {
    "cybercrime_intrusion": ["cybersecurity_misuse"],
    "chemical_biological": ["illegal_criminal"],
    "illegal": ["illegal_criminal", "fraud_scam", "privacy_personal_data"],
    "harmful": [
        "discrimination_toxicity",
        "sexual_graphic",
        "unsafe_unethical",
        "high_stakes_advice",
        "human_chatbot_harm",
    ],
    "misinformation_disinformation": [
        "malicious_influence",
        "misinformation_reliability",
    ],
    "copyright": [],  # Out of scope for our framework
}


def _map_semantic_to_threats(semantic_category: str) -> list[str]:
    return _HARMBENCH_TO_THREAT.get(semantic_category, [])


# =============================================================================
# CSV and JSONL loading
# =============================================================================


def _load_csv(csv_path: str | Path) -> list[dict]:
    """
    Read a HarmBench CSV file and return raw row dicts.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of dicts with raw column values.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"HarmBench CSV not found: {path}\n"
            "Expected columns: Behavior,FunctionalCategory,SemanticCategory,"
            "Tags,ContextString,BehaviorID"
        )
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    logger.debug("Loaded %d rows from '%s'", len(rows), path)
    return rows


def _load_jsonl(jsonl_path: str | Path) -> list[dict]:
    """
    Read a HarmBench JSONL file and return raw row dicts.

    Args:
        jsonl_path: Path to the JSONL file.

    Returns:
        List of dicts with raw field values.

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"HarmBench JSONL not found: {path}\n"
            "Expected fields: Behavior, FunctionalCategory, SemanticCategory, "
            "Tags, ContextString, BehaviorID"
        )
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rows.append(row)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse JSONL line %d: %s", line_num, e)
    logger.debug("Loaded %d rows from '%s'", len(rows), path)
    return rows


def _load_data_file(path: str | Path) -> list[dict]:
    """
    Load a data file (CSV or JSONL) and return raw row dicts.
    Detects format by file extension.

    Args:
        path: Path to the data file (.csv or .jsonl).

    Returns:
        List of dicts with raw field values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not recognized.
    """
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    elif path.suffix.lower() == ".csv":
        return _load_csv(path)
    else:
        raise ValueError(
            f"Unrecognized file extension '{path.suffix}'. "
            "Expected .csv or .jsonl."
        )


def _row_to_behavior_dict(row: dict) -> dict:
    """Convert a raw CSV row to our canonical behavior dict format."""
    semantic_cat = row.get(COL_SEMANTIC, "").strip()
    threats = _map_semantic_to_threats(semantic_cat)
    return {
        "behavior_id": row.get(COL_BEHAVIOR_ID, "").strip(),
        "behavior": row.get(COL_BEHAVIOR, "").strip(),
        "functional_category": row.get(COL_FUNCTIONAL, "standard").strip(),
        "semantic_category": semantic_cat,
        "context_string": row.get(COL_CONTEXT, "").strip(),
        "tags": row.get(COL_TAGS, "").strip(),
        "threat_categories": threats,
        "primary_threat": threats[0] if threats else None,
        "source": "harmbench",
    }


# =============================================================================
# Public API
# =============================================================================


def load_harmbench_behaviors(
    csv_path: str | Path | None = None,
    functional_filter: str = "standard",
    threat_filter: str | None = None,
    exclude_empty: bool = True,
) -> list[dict]:
    """
    Load HarmBench behavior entries and apply filters.

    Args:
        csv_path: Path to HarmBench CSV.  Defaults to HARMBENCH_TEXT_ALL,
            resolved relative to the project root (CWD).
        functional_filter: Which functional categories to include:
            - "standard": only standard behaviors (no extra context required).
            - "contextual": only contextual behaviors.
            - "all": no filter by functional category.
        threat_filter: If set, only include behaviors whose primary_threat or
            threat_categories contains this threat ID string
            (e.g. "cybersecurity_misuse").
        exclude_empty: If True, skip rows with empty Behavior or BehaviorID.

    Returns:
        List of behavior dicts matching the filters.
    """
    if csv_path is None:
        # Try relative to CWD (project root), then module-relative fallback
        candidate = Path.cwd() / HARMBENCH_TEXT_ALL
        if not candidate.exists():
            # Try alternative path in corpora
            alt = (
                Path.cwd()
                / "corpora"
                / "downloaded_datasets"
                / "HarmBench"
                / "data"
                / "behavior_datasets"
                / "harmbench_behaviors_text_all.csv"
            )
            if alt.exists():
                candidate = alt
        csv_path = candidate

    raw_rows = _load_data_file(csv_path)
    behaviors: list[dict] = []

    for row in raw_rows:
        bd = _row_to_behavior_dict(row)

        # Filter empty behaviors
        if exclude_empty and (not bd["behavior"] or not bd["behavior_id"]):
            continue

        # Functional category filter
        if functional_filter != "all":
            if bd["functional_category"] != functional_filter:
                continue

        # Threat category filter
        if threat_filter is not None:
            if (
                threat_filter not in bd["threat_categories"]
                and bd["primary_threat"] != threat_filter
            ):
                continue

        behaviors.append(bd)

    logger.info(
        "Loaded %d behavior(s) from HarmBench "
        "(functional_filter=%s, threat_filter=%s).",
        len(behaviors),
        functional_filter,
        threat_filter,
    )
    return behaviors


def load_harmbench_split(
    split: str = "all",
    functional_filter: str = "standard",
    threat_filter: str | None = None,
) -> list[dict]:
    """
    Load a named HarmBench split.

    Args:
        split: One of "all", "test", "val".
        functional_filter: Functional category filter.
        threat_filter: Optional threat category filter.

    Returns:
        Filtered list of behavior dicts.

    Raises:
        ValueError: If an unknown split name is given.
    """
    split_paths = {
        "all": HARMBENCH_TEXT_ALL,
        "test": HARMBENCH_TEXT_TEST,
        "val": "data/benchmark/harmbench_behaviors_val.jsonl",
    }
    if split not in split_paths:
        raise ValueError(f"Unknown split '{split}'. Choose from: {list(split_paths)}")
    return load_harmbench_behaviors(
        csv_path=Path.cwd() / split_paths[split],
        functional_filter=functional_filter,
        threat_filter=threat_filter,
    )


def get_behavior_texts(behaviors: list[dict]) -> list[str]:
    """Extract the raw behavior strings from a list of behavior dicts."""
    return [b["behavior"] for b in behaviors]


def get_behavior_ids(behaviors: list[dict]) -> list[str]:
    """Extract behavior IDs from a list of behavior dicts."""
    return [b["behavior_id"] for b in behaviors]


def describe_dataset(behaviors: list[dict]) -> dict:
    """
    Return a summary statistics dict about a loaded behavior list.

    Args:
        behaviors: List of behavior dicts.

    Returns:
        Dict with counts by semantic_category, threat, and functional_category.
    """
    from collections import Counter

    sem_counts: Counter = Counter(b["semantic_category"] for b in behaviors)
    threat_counts: Counter = Counter(b["primary_threat"] for b in behaviors)
    func_counts: Counter = Counter(b["functional_category"] for b in behaviors)

    return {
        "total": len(behaviors),
        "by_semantic_category": dict(sem_counts),
        "by_primary_threat": dict(threat_counts),
        "by_functional_category": dict(func_counts),
    }
