"""
ap_generator.py — Attack Prompt (AP) generator.

Takes Base Risk Prompts (BRPs) and generates Attack Prompts (APs) by applying
a specified jailbreak attack technique via the MetaPromptBuilder.

Pipeline:
    BRP (direct harmful request)
      + attack_type (technique to apply)
      → AP (jailbreak-wrapped attack prompt)

Usage:
    python ap_generator.py \\
        --input data/brp_all.jsonl \\
        --attack scenario_nesting \\
        --output data/ap_scenario_nesting.jsonl

    # Apply all 8 attack types to every BRP
    python ap_generator.py --input data/brp_all.jsonl --all-attacks --output data/ap_all.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

_HERE = Path(__file__).parent

from generator.meta_prompt.builder import MetaPromptBuilder  # noqa: E402
from taxonomy.attack_types import ATTACK_TYPES  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AP Generator
# ---------------------------------------------------------------------------


class APGenerator:
    """
    Generates Attack Prompts from Base Risk Prompts.

    Args:
        use_vllm: If True, call vLLM to generate APs (higher quality).
                  If False, render from Jinja2 templates (faster, no API cost).
        vllm_model: vLLM model name.
    """

    def __init__(
        self,
        use_vllm: bool = False,
        vllm_model: str = "qwen-72b",
    ) -> None:
        self.builder = MetaPromptBuilder(
            use_vllm=use_vllm,
            vllm_model=vllm_model,
        )

    def generate_from_brp(
        self,
        brp_records: list[dict],
        attack_type: str,
    ) -> list[dict]:
        """
        Apply a single attack type to all BRP records.

        Args:
            brp_records: List of BRP dicts with keys: id, threat_category, prompt.
            attack_type: Key from ATTACK_TYPES.

        Returns:
            List of AP records with keys: id, threat_category, attack_type,
            brp_id, behavior, prompt, source.
        """
        if attack_type not in ATTACK_TYPES:
            raise ValueError(f"Unknown attack_type: {attack_type!r}")

        ap_records: list[dict] = []
        failed = 0

        for brp in brp_records:
            try:
                meta = self.builder.build(
                    threat_category=brp["threat_category"],
                    attack_type=attack_type,
                    malicious_intent=brp["prompt"],
                )
                ap_records.append(
                    {
                        "id": str(uuid.uuid4()),
                        "threat_category": brp["threat_category"],
                        "attack_type": attack_type,
                        "brp_id": brp.get("id", ""),
                        "behavior": brp["prompt"],  # the underlying intent
                        "prompt": meta.attack_prompt,  # the wrapped attack
                        "template_used": meta.template_used,
                        "source": "ap_generator",
                    }
                )
            except Exception as exc:
                logger.warning("Failed for BRP %s: %s", brp.get("id", "?"), exc)
                failed += 1

        logger.info(
            "Generated %d APs (failed %d) for attack_type='%s'.",
            len(ap_records),
            failed,
            attack_type,
        )
        return ap_records

    def generate_all_attacks(self, brp_records: list[dict]) -> list[dict]:
        """Apply all 8 attack types to every BRP."""
        all_ap: list[dict] = []
        for attack_type in ATTACK_TYPES:
            aps = self.generate_from_brp(brp_records, attack_type)
            all_ap.extend(aps)
        logger.info("Total APs generated: %d", len(all_ap))
        return all_ap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Attack Prompts from BRPs.")
    p.add_argument("--input", type=Path, required=True, help="BRP JSONL input file")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--attack", help="Single attack type key")
    group.add_argument(
        "--all-attacks", action="store_true", help="Apply all 8 attack types"
    )
    p.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    p.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM to generate APs instead of templates",
    )
    p.add_argument("--vllm-model", default="qwen-72b", help="vLLM model name")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    # Load BRPs
    brp_records: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                brp_records.append(json.loads(line))
    logger.info("Loaded %d BRP records from %s", len(brp_records), args.input)

    gen = APGenerator(use_vllm=args.use_vllm, vllm_model=args.vllm_model)

    if args.all_attacks:
        ap_records = gen.generate_all_attacks(brp_records)
    else:
        ap_records = gen.generate_from_brp(brp_records, attack_type=args.attack)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in ap_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Saved %d AP records to %s", len(ap_records), args.output)


if __name__ == "__main__":
    main()
