"""
build_comparison_datasets.py - Build comparison benchmark datasets for benchmark/data/

Converts the following datasets into unified JSONL format:
  1. AdvBench (harmful_behaviors + harmful_strings)
  2. HH-RLHF red-teaming
  3. Flames (Chinese safety dataset)
  4. SafetyPrompts (instruction attack scenarios)
  5. DecodingTrust (harmfulness + toxicity prompts)
  6. EasyJailbreak (download from HuggingFace)

Output format per line:
  {
    "id":              str,
    "behavior":        str,
    "target":          str | None,
    "threat_category": str | None,
    "attack_type":     str | None,
    "source":          str,
    "tags":            list[str]
  }

Usage:
  cd benchmark && python build_comparison_datasets.py [--output_dir data/]
"""

import csv
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOADED_DIR = BASE_DIR / "corpora" / "downloaded_datasets"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"

THREAT_KEYWORDS = {
    "cybersecurity_misuse": ["hack", "exploit", "virus", "malware", "phishing", "sql injection", "xss", "network", "database", "password", "cyber"],
    "illegal_criminal": ["bomb", "weapon", "drug", "steal", "murder", "kill", "illegal", "criminal", "assassinate"],
    "unsafe_unethical": ["dangerous", "unsafe", "unethical", "violent", "harm", "manipulat", "trick", "deceive"],
    "fraud_scam": ["fraud", "scam", "social engineer", "phish", "identity theft"],
    "malicious_influence": ["misinformation", "fake news", "propaganda", "manipulate opinion", "disinformation"],
    "discrimination_toxicity": ["racism", "discrimination", "toxic", "hate", "minority", "stereotype", "bias"],
    "privacy_personal_data": ["privacy", "personal data", "dox", "leak personal"],
    "sexual_graphic": ["sexual", "porn", "graphic", "nsfw"],
    "high_stakes_advice": ["high-stakes", "medical advice", "legal advice", "financial advice"],
    "misinformation_reliability": ["malicious influence", "manipulation", "propaganda"],
    "misinformation_reliability": ["misinformation", "fake", "unreliable", "false claim"],
    "HC": ["chatbot harm", "emotional manipulation", "dependency"],
}

ATTACK_TYPE_KEYWORDS = {
    "scenario_nesting": ["role play", "imagine", "fictional", "story", "hypothetical", "novel", "movie", "game"],
    "persuasion_deception": ["convince", "persuade", "trick", "deceive", "manipulate"],
    "prefix_code_injection": ["write a script", "code", "program", "function", "api"],
    "obfuscation_encryption": ["encode", "cipher", "obfuscat", "base64"],
    "compositional_hybrid": ["combine", "mix", "both"],
}


def map_threat_category(text: str) -> str | None:
    text_lower = text.lower()
    for code, keywords in THREAT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return code
    return None


def map_attack_type(text: str) -> str | None:
    text_lower = text.lower()
    for atype, keywords in ATTACK_TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return atype
    return None


def make_id(source: str, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{source}_{h}"


def convert_advbench(output_dir: Path) -> list[dict]:
    advbench_dir = DOWNLOADED_DIR / "AdvBench" / "data" / "advbench"
    records: list[dict] = []

    behaviors_file = advbench_dir / "harmful_behaviors.csv"
    if behaviors_file.exists():
        with open(behaviors_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                goal = row.get("goal", "").strip()
                target = row.get("target", "").strip()
                if not goal:
                    continue
                records.append({
                    "id": make_id("advbench", goal),
                    "behavior": goal,
                    "target": target,
                    "threat_category": map_threat_category(goal),
                    "attack_type": map_attack_type(goal),
                    "source": "advbench",
                    "tags": ["harmful_behavior"],
                })
        print(f"  AdvBench harmful_behaviors: {len(records)} records")

    strings_file = advbench_dir / "harmful_strings.csv"
    if strings_file.exists():
        count_before = len(records)
        with open(strings_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("target", "").strip()
                if not text:
                    continue
                records.append({
                    "id": make_id("advbench_str", text),
                    "behavior": text,
                    "target": None,
                    "threat_category": map_threat_category(text),
                    "attack_type": "prefix_code_injection",
                    "source": "advbench",
                    "tags": ["harmful_string"],
                })
        added = len(records) - count_before
        print(f"  AdvBench harmful_strings: +{added} records")

    return records


def convert_hh_rlhf_redteam(output_dir: Path) -> list[dict]:
    redteam_file = DOWNLOADED_DIR / "hh-rlhf" / "red-team-attempts" / "red_team_attempts.jsonl.gz"
    records: list[dict] = []

    if not redteam_file.exists():
        print(f"  WARNING: {redteam_file} not found, skipping.")
        return records

    with gzip.open(redteam_file, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            transcript = obj.get("transcript", "")
            if not transcript:
                continue

            parts = transcript.split("\n\nHuman: ")
            if len(parts) > 1:
                first_human = parts[1].split("\n\nAssistant: ")[0].strip()
            else:
                first_human = transcript.strip()

            if not first_human:
                continue

            records.append({
                "id": make_id("hh_rlhf_rt", transcript[:200]),
                "behavior": first_human,
                "target": None,
                "threat_category": map_threat_category(first_human),
                "attack_type": map_attack_type(first_human),
                "source": "hh_rlhf_redteaming",
                "tags": ["red_team", "multi_turn"],
                "full_transcript": transcript if len(transcript) < 5000 else transcript[:5000] + "...",
            })

    print(f"  HH-RLHF red-teaming: {len(records)} records")
    return records


def convert_flames(output_dir: Path) -> list[dict]:
    flames_file = DOWNLOADED_DIR / "Flames" / "Flames_1k_Chinese.jsonl"
    records: list[dict] = []

    if not flames_file.exists():
        print(f"  WARNING: {flames_file} not found, skipping.")
        return records

    with open(flames_file, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = obj.get("prompt", "").strip()
            if not prompt:
                continue

            dimension = obj.get("dimension", "")
            subcomponent = obj.get("subcomponent", "")

            records.append({
                "id": make_id("flames", prompt),
                "behavior": prompt,
                "target": None,
                "threat_category": map_threat_category(prompt),
                "attack_type": map_attack_type(prompt),
                "source": "flames",
                "tags": ["chinese", "safety", f"dim:{dimension}", f"sub:{subcomponent}"],
            })

    print(f"  Flames: {len(records)} records")
    return records


def convert_safety_prompts(output_dir: Path) -> list[dict]:
    sp_file = DOWNLOADED_DIR / "SafetyPrompts" / "instruction_attack_scenarios.json"
    records: list[dict] = []

    if not sp_file.exists():
        print(f"  WARNING: {sp_file} not found, skipping.")
        return records

    with open(sp_file, encoding="utf-8") as f:
        data = json.load(f)

    for category, items in data.items():
        if not isinstance(items, list):
            continue
        for item in items:
            prompt = item.get("prompt", "").strip()
            response = item.get("response", "").strip()
            if not prompt:
                continue

            records.append({
                "id": make_id("safety_prompts", prompt),
                "behavior": prompt,
                "target": response if response else None,
                "threat_category": map_threat_category(prompt),
                "attack_type": map_attack_type(prompt),
                "source": "safety_prompts",
                "tags": ["chinese", f"category:{category}"],
            })

    print(f"  SafetyPrompts: {len(records)} records")
    return records


def convert_decodingtrust(output_dir: Path) -> list[dict]:
    dt_dir = DOWNLOADED_DIR / "DecodingTrust" / "data"
    records: list[dict] = []

    harm_dir = dt_dir / "harmfulness"
    if harm_dir.exists():
        for jsonl_file in harm_dir.rglob("*.jsonl"):
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    prompt = obj.get("prompt", obj.get("text", obj.get("input", ""))).strip()
                    if not prompt:
                        continue

                    rel_path = jsonl_file.relative_to(dt_dir)
                    records.append({
                        "id": make_id("decodingtrust", prompt),
                        "behavior": prompt,
                        "target": None,
                        "threat_category": map_threat_category(prompt),
                        "attack_type": map_attack_type(prompt),
                        "source": "decodingtrust",
                        "tags": ["harmfulness", f"path:{rel_path}"],
                    })

    for tox_file in (dt_dir / "toxicity").rglob("*.jsonl"):
        with open(tox_file, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                prompt_text = obj.get("prompt", {})
                if isinstance(prompt_text, dict):
                    text = prompt_text.get("text", "")
                else:
                    text = str(prompt_text)

                if not text:
                    continue

                records.append({
                    "id": make_id("decodingtrust_tox", text),
                    "behavior": text,
                    "target": None,
                    "threat_category": "discrimination_toxicity",
                    "attack_type": map_attack_type(text),
                    "source": "decodingtrust",
                    "tags": ["toxicity", f"path:{tox_file.relative_to(dt_dir)}"],
                })

    print(f"  DecodingTrust: {len(records)} records")
    return records


def convert_easyjailbreak(output_dir: Path) -> list[dict]:
    records: list[dict] = []

    if not HAS_DATASETS:
        print("  WARNING: 'datasets' library not installed. Skipping EasyJailbreak.")
        return records

    try:
        ds = load_dataset("EasyJailbreak/EasyJailbreak", "AdvBench")
        for split in ds:
            for item in ds[split]:
                query = item.get("query", item.get("prompt", item.get("question", "")))
                if not query:
                    continue

                records.append({
                    "id": make_id("easyjailbreak", query),
                    "behavior": query,
                    "target": item.get("target", item.get("answer", None)),
                    "threat_category": map_threat_category(query),
                    "attack_type": map_attack_type(query),
                    "source": "easyjailbreak",
                    "tags": [f"split:{split}", "subset:AdvBench"],
                })
    except Exception as e:
        print(f"  WARNING: Failed to load EasyJailbreak: {e}")
        try:
            ds = load_dataset("Lemhf11/EasyJailbreak_Datasets")
            for split in ds:
                for item in ds[split]:
                    query = item.get("query", item.get("prompt", ""))
                    if not query:
                        continue

                    records.append({
                        "id": make_id("easyjailbreak", query),
                        "behavior": query,
                        "target": None,
                        "threat_category": map_threat_category(query),
                        "attack_type": map_attack_type(query),
                        "source": "easyjailbreak",
                        "tags": [f"split:{split}"],
                    })
        except Exception as e2:
            print(f"  WARNING: Also failed with alternative: {e2}")
            return records

    print(f"  EasyJailbreak: {len(records)} records")
    return records


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build comparison benchmark datasets")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for JSONL files")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to convert (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converters = {
        "advbench": convert_advbench,
        "hh_rlhf": convert_hh_rlhf_redteam,
        "flames": convert_flames,
        "safety_prompts": convert_safety_prompts,
        "decodingtrust": convert_decodingtrust,
        "easyjailbreak": convert_easyjailbreak,
    }

    if args.datasets:
        converters = {k: v for k, v in converters.items() if k in args.datasets}

    all_records: list[dict] = []

    for name, converter in converters.items():
        print(f"\n{'='*60}")
        print(f"Converting: {name}")
        print("=" * 60)
        records = converter(output_dir)
        all_records.extend(records)

    print(f"\n{'='*60}")
    print("Writing output files...")
    print("=" * 60)

    by_source: dict[str, list[dict]] = {}
    for rec in all_records:
        by_source.setdefault(rec["source"], []).append(rec)

    for source, records in by_source.items():
        out_file = output_dir / f"{source}_behaviors.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {out_file}: {len(records)} records")

    combined_file = output_dir / "all_comparison_behaviors.jsonl"
    with open(combined_file, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n  Combined: {combined_file}: {len(all_records)} total records")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    for source, records in sorted(by_source.items()):
        print(f"  {source:30s}: {len(records):>6d} records")
    print(f"  {'TOTAL':30s}: {len(all_records):>6d} records")


if __name__ == "__main__":
    main()
