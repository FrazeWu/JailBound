#!/usr/bin/env python3
"""
build_behaviors.py — Build benchmark/data/harmbench_behaviors.jsonl

Combines three datasets:
  1. HarmBench text behaviors (standard + contextual, skip copyright)
  2. AdvBench harmful behaviors
  3. StrongREJECT dataset

Output schema per line:
    {
      "id":              str,           # unique identifier
      "behavior":        str,           # the harmful behavior description
      "target":          str,           # expected start of compliance response
      "threat_category": str,           # one of 12 short keys (e.g. "cybersecurity_misuse", "illegal_criminal")
      "attack_type":     str,           # one of 8 attack type keys
      "source":          str,           # "harmbench" | "advbench" | "strongreject"
      "tags":            list[str],     # any extra tags
    }
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
HARMBENCH_CSV = (
    ROOT
    / "corpora/downloaded_datasets/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
HARMBENCH_TARGETS = (
    ROOT
    / "corpora/downloaded_datasets/HarmBench/data/optimizer_targets/harmbench_targets_text.json"
)
ADVBENCH_CSV = (
    ROOT
    / "corpora/downloaded_datasets/AdvBench/data/advbench/harmful_behaviors.csv"
)
STRONGREJECT_CSV = (
    ROOT
    / "corpora/downloaded_datasets/StrongREJECT/strongreject_dataset/strongreject_dataset.csv"
)
OUT_FILE = Path(__file__).parent / "data/harmbench_behaviors.jsonl"

# ---------------------------------------------------------------------------
# HarmBench SemanticCategory → threat_category abbreviation
# ---------------------------------------------------------------------------
HARMBENCH_TO_THREAT: dict[str, str] = {
    "cybercrime_intrusion": "cybersecurity_misuse",
    "chemical_biological": "illegal_criminal",
    "illegal": "illegal_criminal",
    "harmful": "unsafe_unethical",
    "harassment_bullying": "discrimination_toxicity",
    "misinformation_disinformation": "malicious_influence",
    "copyright": None,  # skip
}

# ---------------------------------------------------------------------------
# StrongREJECT category → threat_category abbreviation
# ---------------------------------------------------------------------------
STRONGREJECT_TO_THREAT: dict[str, str] = {
    "Hate, harassment and discrimination": "discrimination_toxicity",
    "Violence": "illegal_criminal",
    "Sexual content": "sexual_graphic",
    "Illegal goods and services": "illegal_criminal",
    "Non-violent crimes": "fraud_scam",
    "Disinformation and deception": "malicious_influence",
}

# ---------------------------------------------------------------------------
# Default attack_type per source/category (heuristic)
# AdvBench and StrongREJECT prompts are mostly "direct harmful instruction"
# → best mapped to "persuasion_deception" (A1) as baseline
# ---------------------------------------------------------------------------
DEFAULT_ATTACK_TYPE = "persuasion_deception"

# HarmBench SemanticCategory → attack type heuristic
HARMBENCH_TO_ATTACK: dict[str, str] = {
    "cybercrime_intrusion": "prefix_code_injection",
    "chemical_biological": "scenario_nesting",
    "illegal": "persuasion_deception",
    "harmful": "contextual_demonstration",
    "harassment_bullying": "persuasion_deception",
    "misinformation_disinformation": "compositional_hybrid",
    "copyright": None,
}


def _uid(text: str, prefix: str = "") -> str:
    """Short deterministic ID from text content."""
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


def build_harmbench() -> list[dict]:
    with open(HARMBENCH_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with open(HARMBENCH_TARGETS, encoding="utf-8") as f:
        targets: dict[str, str] = json.load(f)

    records: list[dict] = []
    for row in rows:
        sem = row["SemanticCategory"]
        func = row["FunctionalCategory"]

        # skip copyright, contextual (need additional context string), multimodal
        if HARMBENCH_TO_THREAT.get(sem) is None:
            continue
        if func == "contextual" and not row.get("ContextString", "").strip():
            continue

        behavior = row["Behavior"].strip()
        if not behavior:
            continue

        bid = row.get("BehaviorID", "").strip()
        target = targets.get(bid, f"Sure, here is how to {behavior[:60].lower()}")

        # If contextual, prepend the context string
        if func == "contextual" and row.get("ContextString", "").strip():
            behavior = row["ContextString"].strip() + "\n\n" + behavior

        tags = [t.strip() for t in row.get("Tags", "").split(",") if t.strip()]

        records.append(
            {
                "id": bid if bid else _uid(behavior, "hb"),
                "behavior": behavior,
                "target": target,
                "threat_category": HARMBENCH_TO_THREAT[sem],
                "attack_type": HARMBENCH_TO_ATTACK.get(sem, DEFAULT_ATTACK_TYPE),
                "source": "harmbench",
                "tags": tags,
            }
        )

    return records


def build_advbench() -> list[dict]:
    with open(ADVBENCH_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records: list[dict] = []
    for row in rows:
        behavior = row.get("goal", "").strip()
        target = row.get("target", "").strip()
        if not behavior:
            continue

        # Heuristic threat category from keywords
        bl = behavior.lower()
        if any(
            k in bl
            for k in [
                "hack",
                "exploit",
                "malware",
                "phish",
                "virus",
                "ransomware",
                "ddos",
                "vuln",
            ]
        ):
            threat = "cybersecurity_misuse"
            attack = "prefix_code_injection"
        elif any(
            k in bl for k in ["weapon", "bomb", "explos", "poison", "drug", "synthesiz"]
        ):
            threat = "illegal_criminal"
            attack = "scenario_nesting"
        elif any(k in bl for k in ["personal", "private", "doxx", "track", "surveil"]):
            threat = "privacy_personal_data"
            attack = "persuasion_deception"
        elif any(
            k in bl
            for k in ["manipulat", "propagand", "misinform", "disinform", "fake news"]
        ):
            threat = "malicious_influence"
            attack = "compositional_hybrid"
        elif any(k in bl for k in ["sexual", "pornograph", "explicit", "nude"]):
            threat = "sexual_graphic"
            attack = "scenario_nesting"
        else:
            threat = "unsafe_unethical"
            attack = DEFAULT_ATTACK_TYPE

        if not target:
            target = f"Sure, here is {behavior[:60].lower()}"

        records.append(
            {
                "id": _uid(behavior, "ab"),
                "behavior": behavior,
                "target": target,
                "threat_category": threat,
                "attack_type": attack,
                "source": "advbench",
                "tags": [],
            }
        )

    return records


def build_strongreject() -> list[dict]:
    with open(STRONGREJECT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records: list[dict] = []
    for row in rows:
        behavior = row.get("forbidden_prompt", "").strip()
        category = row.get("category", "").strip()
        if not behavior:
            continue

        threat = STRONGREJECT_TO_THREAT.get(category, "unsafe_unethical")

        # Attack type heuristic from category
        if category == "Disinformation and deception":
            attack = "compositional_hybrid"
        elif category == "Sexual content":
            attack = "scenario_nesting"
        else:
            attack = DEFAULT_ATTACK_TYPE

        records.append(
            {
                "id": _uid(behavior, "sr"),
                "behavior": behavior,
                "target": f"Sure, here is how to {behavior[:60].lower()}",
                "threat_category": threat,
                "attack_type": attack,
                "source": "strongreject",
                "tags": [category],
            }
        )

    return records


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove near-duplicate behaviors (same first 80 chars after normalization)."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        key = re.sub(r"\s+", " ", r["behavior"][:80].lower().strip())
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Loading HarmBench...", end=" ", flush=True)
    hb = build_harmbench()
    print(f"{len(hb)} records")

    print("Loading AdvBench...", end=" ", flush=True)
    ab = build_advbench()
    print(f"{len(ab)} records")

    print("Loading StrongREJECT...", end=" ", flush=True)
    sr = build_strongreject()
    print(f"{len(sr)} records")

    all_records = hb + ab + sr
    print(f"Total before dedup: {len(all_records)}")

    deduped = deduplicate(all_records)
    print(f"Total after dedup:  {len(deduped)}")

    # Summary by threat category
    from collections import Counter

    threat_counts = Counter(r["threat_category"] for r in deduped)
    attack_counts = Counter(r["attack_type"] for r in deduped)
    source_counts = Counter(r["source"] for r in deduped)

    print("\n=== By threat_category ===")
    for k, v in sorted(threat_counts.items()):
        print(f"  {k:6s}: {v}")

    print("\n=== By attack_type ===")
    for k, v in sorted(attack_counts.items()):
        print(f"  {k:30s}: {v}")

    print("\n=== By source ===")
    for k, v in sorted(source_counts.items()):
        print(f"  {k:15s}: {v}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✓ Written {len(deduped)} behaviors to {OUT_FILE}")


if __name__ == "__main__":
    main()
