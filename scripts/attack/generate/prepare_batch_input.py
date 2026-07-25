"""
prepare_batch_input.py
将 harmbench_behaviors.jsonl 转换为 generate_meta_attack.py 所需的 batch 输入格式。

attack_type 字段映射（下划线格式 → 英文攻击类型名称）：
  scenario_nesting        → Scenario Nesting
  persuasion_deception    → Persuasion & Deception
  compositional_hybrid    → Compositional / Hybrid Attacks
  contextual_demonstration→ Contextual Demonstration
  prefix_code_injection   → Prefix Injection

domain 字段：根据 threat_category 推断最合适的 domain
"""

import json
from pathlib import Path

ATTACK_TYPE_MAP: dict[str, str] = {
    "scenario_nesting": "Scenario Nesting",
    "persuasion_deception": "Persuasion & Deception",
    "compositional_hybrid": "Compositional / Hybrid Attacks",
    "contextual_demonstration": "Contextual Demonstration",
    "prefix_code_injection": "Prefix Injection",
    "input_fragmentation": "Input Fragmentation",
    "obfuscation_encryption": "Obfuscation & Encryption",
    "formal_language": "Formal Language Exploitation",
}

# threat_category → most representative domain
THREAT_TO_DOMAIN: dict[str, str] = {
    "human_chatbot_harm": "SOC",  # human-chatbot → content platforms
    "discrimination_toxicity": "UGC",  # discrimination → content moderation
    "sexual_graphic": "UGC",  # sexual/graphic → content moderation
    "privacy_personal_data": "LAW",  # privacy → legal
    "sensitive_org_gov": "LAW",  # sensitive org/gov → legal
    "cybersecurity_misuse": "SOC",  # cybersecurity → security operations
    "illegal_criminal": "LE",  # illegal → law enforcement
    "fraud_scam": "FIN",  # fraud → finance
    "malicious_influence": "UGC",  # disinformation → media
    "misinformation_reliability": "SCI",  # misinformation → science
    "high_stakes_advice": "MED",  # high-stakes advice → healthcare
    "unsafe_unethical": "ENT",  # unethical → enterprise
}


def convert(input_path: Path, output_path: Path) -> int:
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            threat = d.get("threat_category", "unsafe_unethical")
            attack_raw = d.get("attack_type", "persuasion_deception")
            attack = ATTACK_TYPE_MAP.get(attack_raw, "Persuasion & Deception")
            domain = THREAT_TO_DOMAIN.get(threat, "ENT")
            intent = d.get("behavior", "")

            if not intent:
                continue

            record = {
                "id": d.get("id", f"item_{count}"),
                "threat": threat,
                "attack": attack,
                "domain": domain,
                "intent": intent,
                "source_target": d.get("target", ""),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent
        / "benchmark"
        / "data"
        / "harmbench_behaviors.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "batch_input.jsonl",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = convert(args.input, args.output)
    print(f"Converted {n} records → {args.output}")
