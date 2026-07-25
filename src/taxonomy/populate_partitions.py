#!/usr/bin/env python3
"""
populate_partitions.py — Write samples.jsonl into each attack_types/ and
threat_categories/ subdirectory from the master behaviors file + s_eval data.

Run from the repo root:
    python3 corpora/populate_partitions.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent

MASTER = ROOT / "benchmark" / "data" / "harmbench_behaviors.jsonl"
SEVAL = (
    ROOT
    / "attack_model"
    / "s_eval"
    / "output"
    / "reverse_engineered_dataset.jsonl"
)

ATTACK_TYPES_DIR = HERE / "attack_types"
THREAT_CATS_DIR = HERE / "threat_categories"

# ---- s_eval risk_type → threat snake_case key ----
RISK_TO_THREAT: dict[str, str] = {
    "Crimes_and_Illegal_Activities": "illegal_criminal",
    "Cybersecurity": "cybersecurity_misuse",
    "Data_Privacy": "privacy_personal_data",
    "Ethics_and_Morality": "unsafe_unethical",
    "Extremism": "illegal_criminal",
    "Hate_Speech": "discrimination_toxicity",
    "Inappropriate_Suggestions": "high_stakes_advice",
    "Physical_and_Mental_Health": "human_chatbot_harm",
}

# ---- s_eval category → attack_type ----
CATEGORY_TO_ATTACK: dict[str, str] = {
    "positive_induction": "prefix_code_injection",
    "reverse_induction": "persuasion_deception",
    "goal_hijacking": "persuasion_deception",
    "compositional_instructions": "scenario_nesting",
    "instruction_jailbreak": "contextual_demonstration",
    "deepinception": "scenario_nesting",
    "in_context_attack": "contextual_demonstration",
    "chain_of_utterances": "compositional_hybrid",
    "code_injection": "formal_language",
    "instruction_encryption": "obfuscation_encryption",
}

# ---- threat snake_case key → directory name (keys and values are identical) ----
THREAT_TO_DIR: dict[str, str] = {
    "human_chatbot_harm": "human_chatbot_harm",
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
}


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _dedup(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        key = re.sub(r"\s+", " ", r.get("behavior", "")[:80].lower().strip())
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    # ---- Load master (already categorised) ----
    master_records = _load_jsonl(MASTER)
    print(f"Loaded {len(master_records)} records from master behaviors file")

    # ---- Load s_eval (needs remapping) ----
    seval_records_raw = _load_jsonl(SEVAL)
    seval_records: list[dict] = []
    for r in seval_records_raw:
        import ast

        ext = {}
        if isinstance(r.get("ext"), str):
            try:
                ext = ast.literal_eval(r["ext"])
            except Exception:
                pass
        elif isinstance(r.get("ext"), dict):
            ext = r["ext"]

        category = ext.get("category", r.get("category", ""))
        risk_type = r.get("risk_type", "")

        attack_type = CATEGORY_TO_ATTACK.get(category, "persuasion_deception")
        threat_cat = RISK_TO_THREAT.get(risk_type, "unsafe_unethical")

        behavior = r.get("input", r.get("behavior", "")).strip()
        output = r.get("output", "").strip()

        if not behavior:
            continue

        seval_records.append(
            {
                "id": r.get("traceid", ""),
                "behavior": behavior,
                "target": output[:120] if output else "",
                "threat_category": threat_cat,
                "attack_type": attack_type,
                "source": "s_eval",
                "tags": [category] if category else [],
            }
        )

    print(f"Loaded {len(seval_records)} records from s_eval")

    all_records = _dedup(master_records + seval_records)
    print(f"Total after dedup: {len(all_records)}")

    # ---- Partition by attack_type ----
    by_attack: dict[str, list[dict]] = defaultdict(list)
    for r in all_records:
        at = r.get("attack_type", "")
        if at:
            by_attack[at].append(r)

    print("\n=== Populating attack_types/ ===")
    for attack_type, records in sorted(by_attack.items()):
        dir_name = attack_type  # directory names already match the key
        out_path = ATTACK_TYPES_DIR / dir_name / "samples.jsonl"
        _write_jsonl(out_path, records)
        print(
            f"  {dir_name:35s}: {len(records):4d} records → {out_path.relative_to(HERE)}"
        )

    # ---- Partition by threat_category ----
    by_threat: dict[str, list[dict]] = defaultdict(list)
    for r in all_records:
        tc = r.get("threat_category", "")
        if tc and tc in THREAT_TO_DIR:
            by_threat[tc].append(r)

    print("\n=== Populating threat_categories/ ===")
    for threat_key, records in sorted(by_threat.items()):
        dir_name = THREAT_TO_DIR[threat_key]
        out_path = THREAT_CATS_DIR / dir_name / "samples.jsonl"
        _write_jsonl(out_path, records)
        print(
            f"  {threat_key:35s}: {len(records):4d} records → {out_path.relative_to(HERE)}"
        )

    # ---- also add missing threat_categories dirs ----
    for snake_key, dir_name in THREAT_TO_DIR.items():
        d = THREAT_CATS_DIR / dir_name
        d.mkdir(parents=True, exist_ok=True)

    # ---- also add missing attack_types dirs ----
    for attack_type in [
        "persuasion_deception",
        "scenario_nesting",
        "input_fragmentation",
        "contextual_demonstration",
        "obfuscation_encryption",
        "formal_language",
        "prefix_code_injection",
        "compositional_hybrid",
    ]:
        d = ATTACK_TYPES_DIR / attack_type
        d.mkdir(parents=True, exist_ok=True)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
