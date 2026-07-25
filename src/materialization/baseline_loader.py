"""
Baseline Loader & Taxonomy Mapping

Provides unified dataset loading for all baseline benchmarks, maps them to
12 canonical threat categories, and wraps results as AttackState-compatible
records for unified evaluation.

Datasets:
  AdvBench, HH-RLHF red-teaming, SafetyBench, SG-Bench, HarmBench

Domain mapping:
  medical, financial, legal, scientific, cyber (plus "general")
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DOWNLOADED_DIR = (
    Path(__file__).resolve().parent.parent
    / "corpora"
    / "downloaded_datasets"
)

# =====================================================================
# Taxonomy: 12 canonical threat categories
# =====================================================================

CANONICAL_CATEGORIES = [
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
    "human_chatbot_harm",
]

# HarmBench SemanticCategory → canonical threat categories
HARMBENCH_TO_THREAT: dict[str, list[str]] = {
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
    "copyright": [],
}

# SafetyBench category → canonical category
SAFETYBENCH_TO_THREAT: dict[str, str] = {
    "offensiveness": "discrimination_toxicity",
    "unfairness and discrimination": "discrimination_toxicity",
    "physical health": "high_stakes_advice",
    "mental health": "high_stakes_advice",
    "illegal activities": "illegal_criminal",
    "ethics and morality": "unsafe_unethical",
    "privacy and property": "privacy_personal_data",
}

# SG-Bench safety_type → canonical category
SGBENCH_TO_THREAT: dict[str, str] = {
    "bias": "discrimination_toxicity",
    "toxicity": "discrimination_toxicity",
    "illegal_activity": "illegal_criminal",
    "sensitive_topic": "sensitive_org_gov",
    "privacy": "privacy_personal_data",
    "unsafe_opinion": "unsafe_unethical",
    "harmful_content": "unsafe_unethical",
}

# Domain mapping (keyword-based)
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "medical": ["medical", "health", "drug", "patient", "disease", "doctor",
                "hospital", "treatment", "symptom", "diagnosis", "pharmaceutical"],
    "financial": ["financial", "bank", "money", "investment", "stock", "fraud",
                  "credit", "loan", "tax", "insurance", "crypto"],
    "legal": ["legal", "law", "court", "judge", "attorney", "lawsuit",
              "contract", "regulation", "criminal", "sentence"],
    "scientific": ["scientific", "research", "experiment", "chemical", "biological",
                   "nuclear", "weapon", "explosive", "synthesis", "laboratory"],
    "cyber": ["hack", "malware", "exploit", "vulnerability", "phishing",
              "password", "cyber", "ddos", "ransomware", "injection", "sql"],
}


# =====================================================================
# Unified record type
# =====================================================================

def _make_id(source: str, text: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:12]
    return f"{source}_{h}"


def _infer_domain(text: str) -> str:
    text_lower = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return domain
    return "general"


def _map_threat_category(
    source: str,
    raw_category: str | None,
) -> str:
    """Map dataset-specific category to canonical threat category."""
    if raw_category is None:
        return "unsafe_unethical"  # conservative default

    raw_lower = raw_category.lower().strip()

    if source == "harmbench":
        cats = HARMBENCH_TO_THREAT.get(raw_lower, [])
        return cats[0] if cats else "unsafe_unethical"

    if source == "safetybench":
        return SAFETYBENCH_TO_THREAT.get(raw_lower, "unsafe_unethical")

    if source == "sgbench":
        return SGBENCH_TO_THREAT.get(raw_lower, "unsafe_unethical")

    return "unsafe_unethical"


# =====================================================================
# Dataset loaders (return list[dict])
# =====================================================================

def load_advbench(max_samples: int | None = None) -> list[dict[str, Any]]:
    csv_path = DOWNLOADED_DIR / "AdvBench" / "data" / "advbench" / "harmful_behaviors.csv"
    if not csv_path.exists():
        logger.warning("AdvBench not found at %s", csv_path)
        return []
    records = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            goal = row.get("goal", "").strip()
            if not goal:
                continue
            records.append({
                "id": _make_id("advbench", goal),
                "behavior": goal,
                "source": "advbench",
                "threat_category": _map_threat_category("advbench", None),
                "domain": _infer_domain(goal),
            })
    if max_samples and len(records) > max_samples:
        random.seed(42)
        records = random.sample(records, max_samples)
    logger.info("AdvBench: loaded %d behaviors", len(records))
    return records


def load_malicious_instruct(max_samples: int | None = None) -> list[dict[str, Any]]:
    txt_path = DOWNLOADED_DIR / "MaliciousInstruct" / "data" / "MaliciousInstruct.txt"
    if not txt_path.exists():
        logger.warning("MaliciousInstruct not found at %s", txt_path)
        return []
    records = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            behavior = line.strip()
            if not behavior:
                continue
            records.append({
                "id": _make_id("malicious_instruct", behavior),
                "behavior": behavior,
                "source": "malicious_instruct",
                "threat_category": _map_threat_category("malicious_instruct", None),
                "domain": _infer_domain(behavior),
            })
    if max_samples and len(records) > max_samples:
        random.seed(42)
        records = random.sample(records, max_samples)
    logger.info("MaliciousInstruct: loaded %d behaviors", len(records))
    return records


def load_safetybench(max_samples: int | None = None) -> list[dict[str, Any]]:
    json_path = DOWNLOADED_DIR / "SafetyBench" / "opensource_data" / "test_en.json"
    if not json_path.exists():
        logger.warning("SafetyBench not found at %s", json_path)
        return []
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for item in data:
        q = item.get("question", "").strip()
        if not q:
            continue
        raw_cat = item.get("category")
        records.append({
            "id": _make_id("safetybench", q),
            "behavior": q,
            "source": "safetybench",
            "threat_category": _map_threat_category("safetybench", raw_cat),
            "domain": _infer_domain(q),
        })
    if max_samples and len(records) > max_samples:
        random.seed(42)
        records = random.sample(records, max_samples)
    logger.info("SafetyBench: loaded %d behaviors", len(records))
    return records


def load_sgbench(max_samples: int | None = None) -> list[dict[str, Any]]:
    json_path = (
        DOWNLOADED_DIR / "SGBench" / "datasets" / "SG-Bench"
        / "original_query" / "malicious_instruction.json"
    )
    if not json_path.exists():
        logger.warning("SG-Bench not found at %s", json_path)
        return []
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for item in data:
        q = item.get("query", "").strip()
        if not q:
            continue
        raw_cat = item.get("safety_type")
        records.append({
            "id": _make_id("sgbench", q),
            "behavior": q,
            "source": "sgbench",
            "threat_category": _map_threat_category("sgbench", raw_cat),
            "domain": _infer_domain(q),
        })
    if max_samples and len(records) > max_samples:
        random.seed(42)
        records = random.sample(records, max_samples)
    logger.info("SG-Bench: loaded %d behaviors", len(records))
    return records


def load_harmbench(max_samples: int | None = None) -> list[dict[str, Any]]:
    csv_path = (
        DOWNLOADED_DIR / "HarmBench" / "data" / "behavior_datasets"
        / "harmbench_behaviors_text_test.csv"
    )
    if not csv_path.exists():
        logger.warning("HarmBench not found at %s", csv_path)
        return []
    records = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            behavior = row.get("Behavior", "").strip()
            if not behavior:
                continue
            raw_cat = row.get("SemanticCategory")
            records.append({
                "id": row.get("BehaviorID", _make_id("harmbench", behavior)),
                "behavior": behavior,
                "source": "harmbench",
                "threat_category": _map_threat_category("harmbench", raw_cat),
                "domain": _infer_domain(behavior),
            })
    if max_samples and len(records) > max_samples:
        random.seed(42)
        records = random.sample(records, max_samples)
    logger.info("HarmBench: loaded %d behaviors", len(records))
    return records


# =====================================================================
# Unified loader
# =====================================================================

ALL_LOADERS = {
    "advbench": load_advbench,
    "malicious_instruct": load_malicious_instruct,
    "safetybench": load_safetybench,
    "sgbench": load_sgbench,
    "harmbench": load_harmbench,
}


def load_all_baselines(
    max_per_dataset: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load all baseline datasets.

    Returns:
        {source_name: [records]}  where each record has keys:
        id, behavior, source, threat_category, domain
    """
    result = {}
    for name, loader in ALL_LOADERS.items():
        result[name] = loader(max_per_dataset)
    return result
